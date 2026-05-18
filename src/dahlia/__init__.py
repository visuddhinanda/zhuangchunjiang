import logging
import argparse
import tomllib
import signal
import threading
from concurrent import futures
from time import sleep

from sqlalchemy import create_engine

from . import download,align, scan_paragraphs,glossary,extract,scan_content
from .align import llm_align


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="A rbac service(gRPC).")
    parser.add_argument('-c', '--config', default='config.toml')
    parser.add_argument('-s', "--start",  type=int, default=None, help="起始文件编号（如 1 对应 001.html）")
    parser.add_argument('-e', "--end",    type=int, default=None, help="结束文件编号（如 36 对应 036.html）")

    parser.add_argument('-dl', '--download', help="download's data folder")
    parser.add_argument('-sp', '--scan_paragraphs', help="scan_paragraphs's data folder")
    parser.add_argument('-ex', '--extract', help="extract's data folder")
    parser.add_argument('-a', '--align', help="align's data folder")
    parser.add_argument('-g', '--glossary', action='store_true', help="glossary ")
    parser.add_argument('-sc', '--scan_content', help="scan_paragraphs's data folder")

    parser.add_argument('-d', '--debug', action='store_true', help='run on debug mode')
    parser.add_argument('-v', '--verbose', action='version', version='2026.5.8')
    args = parser.parse_args()

    # ── 日志配置 ──────────────────────────────────────────────────────────────────
    logging.basicConfig(
        format='%(asctime)s [%(levelname).1s] %(message)s', 
        level=logging.DEBUG if args.debug else logging.INFO,
        datefmt="%H:%M:%S",
        )
    logger.debug("running on debug mode")

    logger.debug("load configuration from %s", args.config)
    with open(args.config, "rb") as file:
        config = tomllib.load(file)
        db = open_db(config['postgresql'], args.debug)
        llm_align.init(config['llm'])          # ← 加这一行
        if args.download is not None:
            download.launch(args.download,args.start,args.end)
        if args.scan_paragraphs is not None:
            scan_paragraphs.launch(db, args.scan_paragraphs,args.start,args.end)
        if args.scan_content is not None:
            scan_content.launch(args.scan_content)
        if args.align is not None:
            align.launch(db, args.align,args.start,args.end)
        if args.extract is not None:
            extract.launch(args.extract,args.start,args.end)
        if args.glossary:
            glossary.launch()
        logging.info('done.')


# https://docs.sqlalchemy.org/en/20/dialects/postgresql.html#dialect-postgresql-psycopg-connect
# https://docs.sqlalchemy.org/en/20/dialects/mysql.html#module-sqlalchemy.dialects.mysql.mariadbconnector
# https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#module-sqlalchemy.dialects.sqlite.pysqlite
def open_db(config, debug):
    logger.debug("open postgresql://%s@%s:%d/%s",
                 config['user'], config['host'], config['port'], config['db-name'])
    return create_engine(
        f"postgresql+psycopg://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['db-name']}?sslmode=disable", echo=debug)
