#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/21 
# @Author  : YinLuLu
# @File    : api.py
# @Software: PyCharm

from openai import OpenAI


def baichuan4(system_content, user_content):
    client = OpenAI(
        api_key="YOU_API_KEY",
        base_url="https://api.baichuan-ai.com/v1/",
    )

    completion = client.chat.completions.create(
        model="Baichuan4",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        temperature=0.3,
        stream=True,
    )

    content = ""
    for chunk in completion:
        content += chunk.choices[0].delta.content
    return content


def baichuan3(system_content, user_content):
    client = OpenAI(
        api_key="YOU_API_KEY",
        base_url="https://api.baichuan-ai.com/v1/",
    )

    completion = client.chat.completions.create(
        model="Baichuan3-Turbo",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        temperature=0.3,
        stream=True,
    )

    content = ""
    for chunk in completion:
        content += chunk.choices[0].delta.content
    return content


def chatglm4(system_content, user_content):
    client = OpenAI(
        api_key="YOU_API_KEY",
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )

    completion = client.chat.completions.create(
        model="glm-4-plus",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        temperature=0.9,
        stream=True,
        max_tokens=8192
    )

    content = ""
    for chunk in completion:
        delta = chunk.choices[0].delta
        if delta.content:
            content += delta.content

    return content


def chatglmz1(system_content, user_content):
    client = OpenAI(
        api_key="YOU_API_KEY",
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )

    completion = client.chat.completions.create(
        model="glm-z1-air",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        temperature=0.9,
        stream=True,
        max_tokens=8192
    )

    content = ""
    for chunk in completion:
        delta = chunk.choices[0].delta
        if delta.content:
            content += delta.content

    return content


def doubao_16(system_content, user_content):
    client = OpenAI(
        api_key="YOU_API_KEY",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
    response = client.chat.completions.create(
        model="doubao-seed-1.6-thinking-250615",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        max_tokens=8192,
        temperature=0.8,
        stream=True,
    )
    reasoning_content = ""
    content = ""
    for chunk in response:
        if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
            reasoning_content += chunk.choices[0].delta.reasoning_content
        else:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
    return content


def doubao_15(system_content, user_content):
    client = OpenAI(
        api_key="YOU_API_KEY",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
    response = client.chat.completions.create(
        model="doubao-1.5-thinking-pro-250415",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        max_tokens=8192,
        temperature=0.8,
        stream=True,
    )
    reasoning_content = ""
    content = ""
    for chunk in response:
        if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
            reasoning_content += chunk.choices[0].delta.reasoning_content
        else:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
    return content


def deepseek_r1(system_content, user_content):
    client = OpenAI(
        api_key="YOU_API_KEY",
        base_url="https://api.deepseek.com"
    )

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        stream=True,
        max_tokens=8192,
        temperature=0.8,
    )
    reasoning_content = ""
    content = ""

    for chunk in response:
        if chunk.choices[0].delta.reasoning_content:
            reasoning_content += chunk.choices[0].delta.reasoning_content
        else:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
    return content


def deepseek_v3(system_content, user_content):
    client = OpenAI(
        api_key="YOU_API_KEY",
        base_url="https://api.deepseek.com"
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        max_tokens=8192,
        temperature=0.8,
    )

    return response.choices[0].message.content


def qwen_plus(system_content, user_content):
    client = OpenAI(
        api_key="YOU_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen-plus-latest",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        extra_body={"enable_thinking": True},
        stream=True,
        max_tokens=8192,
        temperature=0.8
    )


    answer_content = ""
    is_answering = False

    for chunk in completion:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            if not is_answering:
                is_answering = True
            answer_content += delta.content

    return answer_content


def qwen3(system_content, user_content):
    client = OpenAI(
        api_key="YOU_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen3-235b-a22b",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        extra_body={"enable_thinking": True},
        stream=True,
        max_tokens=8192,
        temperature=0.8
    )

    answer_content = ""
    is_answering = False


    for chunk in completion:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            if not is_answering:
                is_answering = True
            answer_content += delta.content

    return answer_content
