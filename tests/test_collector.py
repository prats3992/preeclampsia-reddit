from types import SimpleNamespace

from preeclampsia_reddit.collection import reddit as reddit_module
from preeclampsia_reddit.collection.reddit import RedditCollector
from preeclampsia_reddit.storage import LocalStorage


def _post(pid, title, selftext=""):
    return SimpleNamespace(id=pid, title=title, selftext=selftext, author="someone",
                           created_utc=1600000000.0, score=5, num_comments=2,
                           url=f"https://redd.it/{pid}", permalink=f"/r/x/comments/{pid}")


def _comment(cid, body):
    return SimpleNamespace(id=cid, body=body, author=None, created_utc=1600000100.0, score=1,
                           parent_id="t3_x")


class FakeSubreddit:
    def __init__(self, posts):
        self.posts = posts

    def top(self, time_filter, limit):
        return self.posts

    def new(self, limit):
        return self.posts[:1]  # overlaps with top() to exercise de-duplication

    def search(self, query, limit, time_filter):
        return self.posts


class FakeReddit:
    def __init__(self, subreddits, comments):
        self._subreddits, self._comments = subreddits, comments

    def subreddit(self, name):
        return FakeSubreddit(self._subreddits[name])

    def submission(self, id):
        return SimpleNamespace(comments=SimpleNamespace(
            replace_more=lambda limit: None, list=lambda: self._comments))


def test_collector_filters_dedupes_and_stores(tmp_path, monkeypatch):
    monkeypatch.setattr(reddit_module, "RATE_LIMIT_SECONDS", 0)
    fake = FakeReddit(
        subreddits={
            "preeclampsia": [_post("d1", "Any post"), _post("d2", "Another post")],
            "BabyBumps": [_post("g1", "Preeclampsia scare"), _post("g2", "Nursery ideas")],
        },
        comments=[_comment("k1", "short"),
                  _comment("k2", "Hang in there, my preeclampsia resolved after delivery " * 2)],
    )
    store = LocalStorage(tmp_path)
    store.upsert_posts([{"id": "d2", "subreddit": "preeclampsia"}])  # already collected

    totals = RedditCollector(fake, store).run(subreddits=["BabyBumps", "preeclampsia"])

    posts = store.load_posts()
    # dedicated sub keeps everything new; broad sub keeps only core-term matches
    assert set(posts) == {"d1", "d2", "g1"}
    # the fake returns the same comment for every post: counted per post, stored once by id
    assert totals == {"posts": 2, "comments": 2}
    assert len(store.load_comments()) == 1
    assert posts["g1"]["permalink"].startswith("https://reddit.com/")
    comment = next(iter(store.load_comments().values()))
    assert comment["id"] == "k2" and comment["author"] == "[deleted]"
    assert len(store.load_runs()) == 1
