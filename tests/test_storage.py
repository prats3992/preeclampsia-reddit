from preeclampsia_reddit.storage import LocalStorage


def test_local_storage_upsert_merge_and_clear(tmp_path):
    store = LocalStorage(tmp_path)
    assert store.load_posts() == {}

    store.upsert_posts([{"id": "a", "subreddit": "x", "title": "one"}])
    store.upsert_posts([{"id": "a", "subreddit": "x", "title": "edited"},
                        {"id": "b", "subreddit": "y", "title": "two"}])
    posts = store.load_posts()
    assert posts["a"]["title"] == "edited" and set(posts) == {"a", "b"}
    assert "stored_at" in posts["a"]
    assert store.post_ids() == {"a", "b"}

    store.record_run({"collected": {"posts": 2}})
    stats = store.stats()
    assert stats["total_posts"] == 2 and stats["posts_by_subreddit"] == {"x": 1, "y": 1}
    assert len(store.load_runs()) == 1

    store.clear()
    assert store.load_posts() == {} and not any(tmp_path.iterdir())


def test_upsert_without_stamp_preserves_record(tmp_path):
    store = LocalStorage(tmp_path)
    store.upsert_comments([{"id": "c", "stored_at": "original"}], stamp=False)
    assert store.load_comments()["c"]["stored_at"] == "original"
