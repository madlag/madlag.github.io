---
layout:	post
title:	"LLMs for Real-time apps: a comparison"
date:	2024-11-04
description: "A practical comparison of Anthropic, Gemini, and OpenAI for real-time conversational apps"
author: francois
featured: true
tags: [AI, LLM, API]
categories: []
image: assets/images/posts/llm-comparison/llm_comparison_cover.png
---
Choose your champion: Anthropic, Gemini, or OpenAI ?


<div style="clear: both;"></div>


# Point of view

This is based on the needs of a real-time audio educational/conversational app: [Mimir](https://apps.apple.com/fr/app/mimir-questions-and-answers/id6511234817?l=en-GB).


The main pain points we have to address:


- **Latency**
    
    We currently have to add text to speech and speech to text on top of LLM latency, so LLM latency budget is reduced.
    
    We stream LLM output and pipe it into text to speech to reduce perceived latency.
    
    The main metric is first sentence latency, as text to speech often need a complete sentence before starting to stream audio. A secondary metric is first token latency, but it’s often not that different as the main waiting time occurs before the stream start: once the stream starts flowing, getting the first full sentence happens quickly after the first token.
    
- **Structured output**
    
     We execute various tasks based on LLM output, we don't want to parse partially broken YAML or JSON if we can avoid (even if LLMs are quite good at creating well-formed structured output).  Several strategies are available to achieve this: JSON mode and tool use.
    
    All providers are using slight variations of JSON Schema to let users constrain tool calls or json-mode outputs.
    

# 1. Anthropic

## Speed

The fastest after Gemini

- 1.2s approximate first sentence latency for sonnet 3.5
- < 1s haiku

## Quality

Excellent for sonnet, and quite good for haiku 3 (3.5 update coming before end of October 2024)

## Features

- Tool with Streaming (several tools can be selected by the model using several selection modes)
- very cheap caching, even for quite short prompts (2048 tokens for sonnet)

## Missing features

- json-mode (but tool is another way to do that)

## Bugs

- When streaming tool output, [large pauses](https://github.com/anthropics/anthropic-sdk-typescript/issues/529) (quite often > 5s ) can occurs between top level fields with sonnet3.5. We currently have to go around this bug as its impact on user experience is major: no one want an agent to pause between two sentences. Anthropic probably do several LLM “calls” in sequence to generate the different fields, using some kind of [guided generation](https://medium.com/@kevalpipalia/towards-efficiently-generating-structured-output-from-language-models-using-guided-generation-part-e552b04af419) and will probably [improve this in a near future](https://github.com/anthropics/anthropic-sdk-typescript/issues/529#issuecomment-2354091612).
- It’s less an issue with haiku3, but because it’s fast anyway.

## Price

[Pricing](https://www.anthropic.com/pricing#anthropic-api) is similar to OpenAI, a bit higher, and caching can reduce dramatically the cost of complex prompts. Claude-3-haiku is much cheaper than the Sonnet version.

## Conclusion

Powerful, well documented, reliable. If it was not for the tool streaming bug it would be perfect.

# Gemini

## Speed

The fastest overall.

gemini-1.5-flash quality is already very good with a < 1s latency for first sentence.

gemini-1.5-flash-8b is blazing fast (< 0.5s median time)

gemini-1.5-pro is decent in terms of speed (~2.5s for first sentence), but much slower than claude-3-5-sonnet .

## Quality

We will need to evaluate it more thoroughly, but the first impression is that quality is good, even for flash or flash-8b.

## Features

- Json mode : specify a large enough subset of JSON schema that constrains the output. Some missing features are `minItems` and `maxItems` constraints on lists, but it's not really an issue for us.
- Tools (several tools can be selected by the model using several selection modes)
- Caching

## Missing features

- Tools streaming: the tools only send their outputs when complete. This is a major issue for us as we want to stream the tool output to process it "on the fly". I could almost flag this as a "bug" or a "major oversight" because it should be quite straightforward for the Gemini team to stream partial results and the impact of not doing so is major on API usability for developers.

## Price

Gemini is in its own category in term of price. gemini-1.5-flash is something like 50x cheaper than the competition. And with caching the difference is even larger.

## Conclusion

As for Anthropic, we have to go around some issues with tool streaming (slow for Anthropic, nonexistent for Gemini), but the speed is so high and the price so low that it opens new possibilities to implement our pipeline.

# OpenAI

## Speed

In terms of latency, gpt-4 and gpt-4-turbo are comparable at roughly 1.5 second for first sentence. gpt-4-turbo is roughly 2x faster overall, but as what is interesting to us is first sentence latency, it’s not that important (but price may be).

gpt-4o is totally unusable for our use case, as latency is more than 5s for first sentence. 

gpt-4o-mini is very fast, at less than 1s of first sentence latency.

Speed is somewhat variable throughout the day, probably due to variable traffic on OpenAI server, so expect some non negligible variability.

## Quality

Once more, we will have to test more extensively, but you get good quality on gpt-4 variants, and decent quality on gpt-4o-mini.

## Features

- Json mode : the full spectrum is available on gpt-4o versions, and significant subset on older version. You can have json without schema or function calls with json schema on all major versions (called json mode on old gpt-4 models), and gpt-4o versions have introduced a json-schema constraint for the output, independently of function calls.
- Caching (half the price, something less interesting than other providers who offer large discounts when using caching).

## Missing features

- No show stoppers for us, but latency is really an issue on gpt-4o

## Price

[Prices](https://openai.com/api/pricing/) are a bit lower than Anthropic, but quite similar.

gpt-4o-mini is really cheap, but not as cheap as gemini-1.5-flash.

## Conclusion

Very strong solution, but they are somewhat victims of their success, as latency is really too high on their latest model gpt-4o and at some points in the day.

# Conclusion

There is not really a 100% winner here.

- Anthropic Claude-3.5-Sonnet is almost perfect, but a bug on streaming tools is a major issue.
- Gemini is very fast and extremely affordable, but lacks support for tool streaming.
- OpenAI supports all you can think of in terms of API features with gpt-4o, but the standard model is too slow for real-time apps. The mini variant is decent in terms of speed but may be less consistent in terms of quality.