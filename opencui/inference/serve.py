#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dataclasses
import getopt
import logging
import json
import sys
from enum import Enum

from aiohttp import web

from opencui.inference.converter import load_converter

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

routes = web.RouteTableDef()


Enum("DugMode", ["SKILL", "SLOT", "BINARY", "SEGMENT"])

# This can be used to serve the whole thing, or just prompt service.


@routes.get("/hello")
async def hello(_: web.Request):  # For heart beat
    return web.Response(text="Hello, world")


@routes.post("/v1/predict")
async def understand(request: web.Request):
    text = await request.text()
    print(text)
    req = json.loads(text)
    logging.info(req)

    utterance = req.get("utterance")

    if len(utterance) == 0:
        return web.json_response({"errMsg": f"empty user input."})

    mode = req.get("mode")
    l_converter: Converter = request.app["converter"]

    if mode == "SEGMENT":
        return web.json_response({"errMsg": f"Not implemented yet."})

    if mode == "SKILL":
        expectations = req.get("expectations")
        results = l_converter.detect_triggerables(utterance, expectations)
        response = [
            {"utterance": utterance, "ownerFrame": func} for func in results
        ]
        return web.json_response(response)

    if mode == "SLOT":
        slots = req.get("slots")
        entities = req.get("entities")
        results = l_converter.fill_slots(utterance, slots, entities)
        return web.json_response(results)

    if mode == "BINARY":
        questions = req.get(questions)
        # So that we can use different llm.
        resp = l_converter.generate(utterance, questions)
        return web.json_response(resp)


def init_app(converter):
    app = web.Application()
    app.add_routes(routes)
    app["converter"] = converter
    return app

def load_converter_from_meta(module_path):
    index_path = f"{module_path}/index/"
    return load_converter(module_path, index_path)

if __name__ == "__main__":
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "hi:s:")
    cmd = False
    for opt, arg in opts:
        if opt == "-h":
            print(
                "serve.py -s <services/agent meta directory, separated by ,> -i <directory for index>"
            )
            sys.exit()
        elif opt == "-s":
            module_path = arg

    converter = load_converter_from_meta(module_path)

    web.run_app(init_app(converter), port=3001)
