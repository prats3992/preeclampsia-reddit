"""Command-line interface (``preeclampsia-reddit`` or ``python -m preeclampsia_reddit``)."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from preeclampsia_reddit.settings import Settings, load_settings
from preeclampsia_reddit.storage import BACKENDS, get_storage

log = logging.getLogger("preeclampsia_reddit")


def cmd_collect(args, settings: Settings) -> None:
    from preeclampsia_reddit.collection.reddit import RedditCollector

    storage = get_storage(settings, args.backend)
    collector = RedditCollector.from_settings(settings, storage)
    collector.run(subreddits=args.subreddits, collect_comments=not args.no_comments)


def cmd_analyze(args, settings: Settings) -> None:
    from preeclampsia_reddit.pipeline import run_analysis

    result = run_analysis(get_storage(settings, args.backend), settings.processed_dir,
                          settings.results_dir)
    log.info("Done: %d posts, %d comments analysed. Report: %s", len(result["posts"]),
             len(result["comments"]), result["report"])


def cmd_sync(args, settings: Settings) -> None:
    local, remote = get_storage(settings, "local"), get_storage(settings, "firebase")
    source, target = (remote, local) if args.direction == "pull" else (local, remote)

    posts, comments = source.load_posts(), source.load_comments()
    # Keep the original stored_at timestamps when copying between backends
    target.upsert_posts(posts.values(), stamp=False)
    target.upsert_comments(comments.values(), stamp=False)
    log.info("Copied %d posts and %d comments from %s to %s", len(posts), len(comments),
             source.name, target.name)


def cmd_stats(args, settings: Settings) -> None:
    print(json.dumps(get_storage(settings, args.backend).stats(), indent=2))


def cmd_clear(args, settings: Settings) -> None:
    storage = get_storage(settings, args.backend)
    if not args.yes:
        answer = input(f"Permanently delete ALL data in {storage.name} storage? Type 'yes': ")
        if answer.strip().lower() != "yes":
            log.info("Aborted.")
            return
    storage.clear()
    log.info("Cleared %s storage.", storage.name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preeclampsia-reddit",
        description="Collect and analyse pre-eclampsia discussions on Reddit.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_backend(p):
        p.add_argument("--backend", choices=BACKENDS,
                       help="storage backend (default: $STORAGE_BACKEND or 'local')")

    p = sub.add_parser("collect", help="collect posts and comments from Reddit")
    add_backend(p)
    p.add_argument("--subreddits", nargs="+", metavar="NAME",
                   help="subset of configured subreddits (default: all)")
    p.add_argument("--no-comments", action="store_true", help="skip comment collection")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("analyze", help="clean, score and analyse data; regenerate results/")
    add_backend(p)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("sync", help="copy data between Firebase and local storage")
    p.add_argument("direction", choices=("pull", "push"),
                   help="pull: Firebase -> local, push: local -> Firebase")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("stats", help="show counts of stored posts and comments")
    add_backend(p)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("clear", help="delete all stored data in a backend")
    add_backend(p)
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.set_defaults(func=cmd_clear)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    try:
        args.func(args, load_settings())
    except RuntimeError as e:  # configuration / missing-data errors with actionable messages
        log.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
