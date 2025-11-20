#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 BeThere AI
# All rights reserved.
#
# This source code is licensed under the BeThere AI license.
# See LICENSE file in the project root for full license information.

import os

from opendu.core.embedding import EmbeddingStore, EmbeddingType
os.environ['HF_HUB_OFFLINE'] = '1'  # Set this FIRST
os.environ["VLLM_DISABLE_TELEMETRY"] = "1"
# Set these FIRST, before any other imports
os.environ["TRITON_DISABLE_LINE_INFO"] = "1"
os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ['VLLM_DISABLE_TRITON_KERNEL'] = '1'


import numpy as np
import getopt
import logging
import sys
from enum import Enum
from lru import LRU
import traceback as tb
from aiohttp import web
import shutil
from opendu.core.config import RauConfig
from opendu.inference.parser import Parser, Decoder, load_parser
from opendu.core.index import indexing
from sentence_transformers import SentenceTransformer



logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

routes = web.RouteTableDef()


Enum("DugMode", ["SKILL", "SLOT", "BINARY", "SEGMENT"])

# This service is designed to provide the indexing and serving for many agents, so that
# We can reduce the startup time.
"""
curl -X POST -d '{"mode":"SKILL","utterance":"make a reservation","expectations":[],"slotMetas":[],"entityValues":{},"questions":[]}' 127.0.0.1:3001/v1/predict/tableReservation
curl -X POST -d '{"mode":"BINARY","utterance":"Yes, absolutely.","questions":["Are you sure you want the white one?"]}' 127.0.0.1:3001/v1/predict/agent
curl -X POST -d '{"mode": "SLOT", "utterance": "order food", "frames": [], "slots": [], "candidates": {}, "dialogActs": []}' http://127.0.0.1:3001/v1/predict/tableReservation
"""


@routes.get("/hello")
async def hello(_: web.Request):  # For heart beat
    return web.Response(text=f"Ok")


@routes.get("/v1/index/{bot}")
async def index(request: web.Request):
    bot = request.match_info["bot"]
    root = request.app["root"]
    bot_path = f"{root}/{bot}"
    index_path = f"{root}/{bot}/index"
    # We remove converter, delete the index, and index again.
    converters = request.app["converters"]

    # Remove the old object.
    converters[bot] = None
    if os.path.exists(index_path):
        logging.info(f"remove index for {bot}")
        shutil.rmtree(index_path)

    logging.info(f"create index for {bot}")
    try:
        indexing(bot_path)

        # Assume it is always a good idea to reload the index.
        reload(bot, request.app)
    except Exception as e:
        traceback_str = "".join(tb.format_exception(None, e, e.__traceback__))
        print(traceback_str)
        return web.Response(text=traceback_str, status=500)

    return web.Response(text="Ok")


@routes.get("/v1/load/{bot}")
async def load(request: web.Request):
    bot = request.match_info["bot"]
    try:
        reload(bot, request.app)
    except Exception as e:
        traceback_str = "".join(tb.format_exception(None, e, e.__traceback__))
        print(traceback_str)
        return web.Response(text=traceback_str, status=500)

    # client will only check 200.
    return web.Response(text="Ok")


@routes.post("/v1/predict/{bot}")
async def understand(request: web.Request):
    bot = request.match_info["bot"]

    # Make sure we have reload the index.
    try:
        reload(bot, request.app)
    except Exception as e:
        traceback_str = "".join(tb.format_exception(None, e, e.__traceback__))
        print(traceback_str)
        return web.Response(text=traceback_str, status=500)

    req = await request.json()
    logging.info(req)

    utterance = req.get("utterance")

    if len(utterance) == 0:
        return web.json_response({"errMsg": f"empty user input."})

    mode = req.get("mode")
    l_converter: Parser = request.app["converters"][bot]

    if mode == "SKILL":
        try:
            expectations = req.get("expectedFrames")
            candidates = req.get("candidates")
            results = l_converter.detect_triggerables(
                utterance, candidates, expectations
            )
        except Exception as e:
            traceback_str = "".join(tb.format_exception(None, e, e.__traceback__))
            print(f"skill detect something is wrong; ${traceback_str}")
            return web.Response(text=traceback_str, status=500)
        return web.json_response(results)

    if mode == "SLOT":
        try:
            frame_name = req.get("targetFrame")
            slots = req.get("slots")
            candidates = req.get("candidates")
            expectedSlots = req.get("expectedSlots")
            results = l_converter.fill_slots(utterance, frame_name, slots, candidates)
            logging.info(results)
        except Exception as e:
            traceback_str = "".join(tb.format_exception(None, e, e.__traceback__))
            print(f"slot fill something is wrong; ${traceback_str}")
            return web.Response(text=traceback_str, status=500)

        return web.json_response(results)

    if mode == "BINARY":
        try:
            question = req.get("question")
            dialog_act_type = req.get("dialogActType")
            target_frame = req.get("targetFrame")
            target_slot = req.get("targetSlot")
            # So that we can use different llm.
            resp = l_converter.inference(
                utterance, question, dialog_act_type, target_frame, target_slot
            )
        except Exception as e:
            traceback_str = "".join(tb.format_exception(None, e, e.__traceback__))
            print(f"yes/no something is wrong; ${traceback_str}")
            return web.Response(text=traceback_str, status=500)

        return web.json_response(resp)


# This reload the converter from current indexing.
def reload(key, app):
    root = app["root"]
    converters = app["converters"]
    bot_path = f"{root}/{key}"
    if key not in converters or converters[key] is None:
        logging.info(f"load index for {key}...")
        index_path = f"{bot_path}/index/"
        converters[key] = load_parser(bot_path, index_path)
        logging.info(f"bot {key} is ready.")


def init_app(schema_root, size):
    app = web.Application()
    app.add_routes(routes)
    app["converters"] = LRU(size)
    app["root"] = schema_root
    return app


if __name__ == "__main__":
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "hi:s:")
    cmd = False
    lru_capacity = 32
    for opt, arg in opts:
        if opt == "-h":
            print("serve.py -s <root for services/agent schema>")
            sys.exit()
        elif opt == "-s":
            root_path = arg
        elif opt == "-i":
            lru_capacity = int(arg)

    # This load the generator LLM first.
    EmbeddingStore.get_embedding_by_task(EmbeddingType.DESC)
    Decoder.get()
    app = init_app(root_path, lru_capacity)

    # Let me write this to file?
    with open("/tmp/rau_ready.flag", "w") as f:
        f.write("ready")
    
    web.run_app(app, port=3001)
