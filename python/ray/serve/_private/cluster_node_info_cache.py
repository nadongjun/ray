import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple, Union

import ray
from ray._raylet import GcsClient
from ray.serve._private.constants import RAY_GCS_RPC_TIMEOUT_S

logger = logging.getLogger(__name__)

CUSTOM_NODE_LABELS_KV_PREFIX = "node-labels-"


class ClusterNodeInfoCache(ABC):
    """Provide access to cached node information in the cluster."""

    def __init__(self, gcs_client: GcsClient, kv_store=None):
        self._gcs_client = gcs_client
        self._kv_store = kv_store
        self._cached_alive_nodes = None
        self._cached_node_labels = dict()
        self._custom_node_labels: Dict[str, Dict[str, str]] = dict()
        self._cached_total_resources_per_node = dict()
        self._cached_available_resources_per_node = dict()

    def update(self):
        """Update the cache by fetching latest node information from GCS.

        This should be called once in each update cycle.
        Within an update cycle, everyone will see the same
        cached node info avoiding any potential issues
        caused by inconsistent node info seen by different components.
        """
        nodes = self._gcs_client.get_all_node_info(timeout=RAY_GCS_RPC_TIMEOUT_S)
        alive_nodes = [
            (node_id.hex(), node.node_name, node.instance_id)
            for (node_id, node) in nodes.items()
            if node.state == ray.core.generated.gcs_pb2.GcsNodeInfo.ALIVE
        ]

        # Sort on NodeID to ensure the ordering is deterministic across the cluster.
        sorted(alive_nodes)
        self._cached_alive_nodes = alive_nodes
        self._cached_node_labels = {
            node_id.hex(): dict(node.labels) for (node_id, node) in nodes.items()
        }

        # Node resources
        self._cached_total_resources_per_node = {
            node_id.hex(): dict(node.resources_total)
            for (node_id, node) in nodes.items()
        }

        self._cached_available_resources_per_node = (
            ray._private.state.available_resources_per_node()
        )

        # Reload custom labels from KV store for alive nodes.
        # This ensures labels survive controller restarts.
        if self._kv_store is not None:
            alive_ids = {node_id for node_id, _, _ in alive_nodes}
            self._reload_custom_labels_from_kv(alive_ids)

    def get_alive_nodes(self) -> List[Tuple[str, str, str]]:
        """Get IDs, IPs, and Instance IDs for all live nodes in the cluster.

        Returns a list of (node_id: str, node_ip: str, instance_id: str).
        The node_id can be passed into the Ray SchedulingPolicy API.
        """
        return self._cached_alive_nodes

    def get_total_resources_per_node(self) -> Dict[str, Dict]:
        """Get total resources for alive nodes."""
        return self._cached_total_resources_per_node

    def get_alive_node_ids(self) -> Set[str]:
        """Get IDs of all live nodes in the cluster."""
        return {node_id for node_id, _, _ in self.get_alive_nodes()}

    @abstractmethod
    def get_draining_nodes(self) -> Dict[str, int]:
        """Get draining nodes in the cluster and their deadlines."""
        raise NotImplementedError

    @abstractmethod
    def get_node_az(self, node_id: str) -> Optional[str]:
        """Get availability zone of a node."""
        raise NotImplementedError

    def get_active_node_ids(self) -> Set[str]:
        """Get IDs of all active nodes in the cluster.

        A node is active if it's schedulable for new tasks and actors.
        """
        return self.get_alive_node_ids() - set(self.get_draining_nodes())

    def get_available_resources_per_node(self) -> Dict[str, Union[float, Dict]]:
        """Get available resources per node.

        Returns a map from (node_id -> Dict of resources).
        """

        return self._cached_available_resources_per_node

    def get_node_labels(self, node_id: str) -> Dict[str, str]:
        """Get the labels for a specific node from the cache.

        Returns GCS labels merged with any custom labels set via
        set_node_labels(). Custom labels take precedence.
        """
        labels = self._cached_node_labels.get(node_id, {}).copy()
        custom = self._custom_node_labels.get(node_id)
        if custom:
            labels.update(custom)
        return labels

    def get_all_node_labels(self) -> Dict[str, Dict[str, str]]:
        """Get merged labels for all alive nodes."""
        return {
            node_id: self.get_node_labels(node_id)
            for node_id in self.get_alive_node_ids()
        }

    def set_node_labels(self, node_id: str, labels: Dict[str, str]) -> None:
        """Dynamically set custom labels for a node.

        Labels are persisted in KV store (if available) so they survive
        controller restarts. Custom labels are merged on top of GCS labels.
        """
        self._custom_node_labels[node_id] = labels.copy()
        if self._kv_store is not None:
            try:
                key = f"{CUSTOM_NODE_LABELS_KV_PREFIX}{node_id}"
                self._kv_store.put(key, json.dumps(labels).encode())
            except Exception:
                logger.warning(
                    f"Failed to persist custom labels for node {node_id} "
                    "to KV store.",
                    exc_info=True,
                )

    def patch_node_labels(self, node_id: str, labels: Dict[str, str]) -> None:
        """Add or update specific custom labels for a node.

        Unlike set_node_labels which replaces all custom labels,
        this merges the given labels into existing custom labels.
        """
        existing = self._custom_node_labels.get(node_id, {}).copy()
        existing.update(labels)
        self._custom_node_labels[node_id] = existing
        if self._kv_store is not None:
            try:
                key = f"{CUSTOM_NODE_LABELS_KV_PREFIX}{node_id}"
                self._kv_store.put(key, json.dumps(existing).encode())
            except Exception:
                logger.warning(
                    f"Failed to persist custom labels for node {node_id} "
                    "to KV store.",
                    exc_info=True,
                )

    def delete_node_label(self, node_id: str, label_key: str) -> None:
        """Delete a specific custom label key for a node.

        If the key doesn't exist in custom labels, this is a no-op (idempotent).
        """
        custom = self._custom_node_labels.get(node_id, {})
        custom.pop(label_key, None)
        self._custom_node_labels[node_id] = custom
        if self._kv_store is not None:
            try:
                key = f"{CUSTOM_NODE_LABELS_KV_PREFIX}{node_id}"
                self._kv_store.put(key, json.dumps(custom).encode())
            except Exception:
                logger.warning(
                    f"Failed to persist custom labels for node {node_id} "
                    "to KV store.",
                    exc_info=True,
                )

    def _reload_custom_labels_from_kv(self, alive_node_ids: Set[str]) -> None:
        """Reload custom labels from KV store for alive nodes."""
        reloaded = {}
        for node_id in alive_node_ids:
            try:
                key = f"{CUSTOM_NODE_LABELS_KV_PREFIX}{node_id}"
                val = self._kv_store.get(key)
                if val is not None:
                    reloaded[node_id] = json.loads(val.decode())
            except Exception:
                logger.debug(
                    f"Failed to load custom labels for node {node_id} "
                    "from KV store.",
                    exc_info=True,
                )
        self._custom_node_labels = reloaded


class DefaultClusterNodeInfoCache(ClusterNodeInfoCache):
    def __init__(self, gcs_client: GcsClient, kv_store=None):
        super().__init__(gcs_client, kv_store=kv_store)

    def get_draining_nodes(self) -> Dict[str, int]:
        return dict()

    def get_node_az(self, node_id: str) -> Optional[str]:
        """Get availability zone of a node."""
        return None
