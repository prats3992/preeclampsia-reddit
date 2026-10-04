"""FirebaseStorage and `sync` against an in-memory stand-in for ``firebase_admin``."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from preeclampsia_reddit import cli
from preeclampsia_reddit.settings import Settings
from preeclampsia_reddit.storage import LocalStorage, get_storage
from preeclampsia_reddit.storage.firebase import UPSERT_CHUNK, FirebaseStorage


class FakeReference:
    """Minimal Realtime Database reference: nested dicts addressed by '/'-separated paths."""

    def __init__(self, tree: dict, path: tuple = ()):
        self.tree, self.path = tree, path
        self.update_calls = 0

    def _node(self, create=False):
        node = self.tree
        for key in self.path:
            if key not in node:
                if not create:
                    return None
                node[key] = {}
            node = node[key]
        return node

    def child(self, path):
        ref = FakeReference(self.tree, self.path + tuple(path.split("/")))
        ref.root = getattr(self, "root", self)
        return ref

    def get(self):
        return self._node()

    def update(self, values):
        getattr(self, "root", self).update_calls += 1
        for path, value in values.items():
            *parents, leaf = path.split("/")
            node = self._node(create=True)
            for key in parents:
                node = node.setdefault(key, {})
            node[leaf] = value

    def delete(self):
        *parents, leaf = self.path
        parent = FakeReference(self.tree, tuple(parents))._node()
        if parent:
            parent.pop(leaf, None)


@pytest.fixture
def fake_firebase(monkeypatch):
    tree: dict = {}
    root = FakeReference(tree)
    state = SimpleNamespace(tree=tree, root=root, init_args=None)

    admin = ModuleType("firebase_admin")
    admin._apps = {}

    def initialize_app(cred, options):
        state.init_args = (cred, options)
        admin._apps["[DEFAULT]"] = object()

    admin.initialize_app = initialize_app
    admin.credentials = SimpleNamespace(Certificate=lambda path: ("cert", path),
                                        ApplicationDefault=lambda: ("adc",))
    admin.db = SimpleNamespace(reference=lambda: root)
    monkeypatch.setitem(sys.modules, "firebase_admin", admin)
    return state


def test_initialisation_uses_credentials_file_and_url(fake_firebase, tmp_path):
    key = tmp_path / "key.json"
    key.write_text("{}")
    FirebaseStorage("https://db.example", key)
    assert fake_firebase.init_args == (("cert", str(key)), {"databaseURL": "https://db.example"})


def test_initialisation_errors_are_actionable(fake_firebase, tmp_path):
    with pytest.raises(RuntimeError, match="FIREBASE_DATABASE_URL"):
        FirebaseStorage(None)
    with pytest.raises(RuntimeError, match="not found"):
        FirebaseStorage("https://db.example", tmp_path / "missing.json")


def test_round_trip_uses_original_node_names(fake_firebase):
    store = FirebaseStorage("https://db.example")
    store.upsert_posts([{"id": "p1", "subreddit": "preeclampsia"}])
    store.upsert_comments([{"id": "c1", "post_id": "p1"}])
    store.record_run({"collected": {"posts": 1}})

    tree = fake_firebase.tree
    assert set(tree) == {"reddit_posts", "reddit_comments", "collection_metadata"}
    assert store.load_posts()["p1"]["subreddit"] == "preeclampsia"
    assert store.stats()["total_comments"] == 1

    tree["llm_comparison_analysis"] = {"x": {}}  # legacy node from older collector versions
    store.clear()
    assert tree == {}
    assert store.load_posts() == {}


def test_large_upserts_are_chunked(fake_firebase):
    store = FirebaseStorage("https://db.example")
    store.upsert_posts({"id": f"p{i}"} for i in range(UPSERT_CHUNK * 2 + 1))
    assert fake_firebase.root.update_calls == 3
    assert len(store.load_posts()) == UPSERT_CHUNK * 2 + 1


def test_array_shaped_nodes_are_normalised(fake_firebase):
    fake_firebase.tree["reddit_posts"] = [{"id": "0"}, None, {"id": "2"}]
    assert set(FirebaseStorage("https://db.example").load_posts()) == {"0", "2"}


def test_sync_pull_and_push(fake_firebase, tmp_path, synthetic_records):
    settings = Settings(data_dir=tmp_path / "data", firebase_database_url="https://db.example")
    posts, comments = synthetic_records
    remote = get_storage(settings, "firebase")
    remote.upsert_posts(posts)
    remote.upsert_comments(comments)
    stamped = remote.load_posts()["p0000"]["stored_at"]

    cli.cmd_sync(SimpleNamespace(direction="pull"), settings)
    local = LocalStorage(settings.raw_dir)
    assert len(local.load_posts()) == 300 and len(local.load_comments()) == 300
    assert local.load_posts()["p0000"]["stored_at"] == stamped  # original timestamp kept

    remote.clear()
    cli.cmd_sync(SimpleNamespace(direction="push"), settings)
    assert len(remote.load_posts()) == 300


def test_analyze_reads_directly_from_firebase(fake_firebase, tmp_path, synthetic_records):
    settings = Settings(data_dir=tmp_path / "data", results_dir=tmp_path / "results",
                        firebase_database_url="https://db.example")
    posts, comments = synthetic_records
    store = get_storage(settings, "firebase")
    store.upsert_posts(posts)
    store.upsert_comments(comments)

    cli.cmd_analyze(SimpleNamespace(backend="firebase"), settings)
    assert (tmp_path / "results" / "report.html").exists()
