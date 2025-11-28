---
layout: post
title: "Apple's AI Future"
date: 2025-11-28
description: "What's Apple AI strategy?"
author: francois
tags: [AI, Infrastructure, Apple, Local models]
categories: []
image: assets/images/posts/apple-ai-future/cover.jpg
featured: true
---

To paraphrase a [famous quote about football](https://en.wikiquote.org/wiki/Gary_Lineker#:~:text=Football%20is%20a%20simple%20game,iNews.) (soccer for our American fellows):
 > "Technology is a simple game, thousands of companies are playing it, and in the end, Apple wins."

On AI, however, one could argue that Apple's odds don't look good.
Apple has not played well *all* the revolutions of the past few years (cars, VR). And you may not have to be able to ride every wave to succeed. But some are taller than others, and AI looks more like a tsunami than a ripple.

# Hardware & Software stacks
It's not that Apple is not trying.

They have been adding specific hardware for AI since 2017, prehistoric times speaking about AI. It's called Neural Engine, and it have been in all iPhones since then. Their M-series chips are also very good for AI with their shared memory architecture, and you can run the largest open source models on them without a lot of effort.
It's something unique in the industry: although a few Google phones include some mobile-specific TPUs, the fragmentation of the Android market makes it hard for developers to consider building something specific for those TPUs, whereas Apple has a unique position with a very coherent line of products.

But it's hard to get developers to use new APIs in real products.
You need a lot of resources to do the research and then even more to make it work in a real product.
Building a proof-of-concept is one thing. Shipping something that actually works at scale is another entire beast. You need to learn the API, rewrite parts of your code, test edge cases, and hope the tooling is mature enough.
I often give as a rule of thumb that you need at least 4x more "D" engineers than "R" engineers, or you will waste good research because you will not be able to use it in products.

(By the way, advice to developers: if you want to get some huge visibility, try to use one of their newly released APIs, [Apple will propel you to the front stage](https://www.youtube.com/watch?v=gZLdhcC8BW0) while advertising their APIs. That's how they will reward you, as most developers, especially in larger companies, will wait until someone else has taken the risk first.)

# Mobile and Cloud AI usage
So even if Apple has the best-in-class hardware and some good APIs to use it, it's still not much used by the developers themselves.

The largest users of these capabilities are the companies themselves: of course Apple is using it for its own OS and software.
For photography, Face ID, AR and VR, Apple has been using the neural engine for a long time, and it gives them a huge advantage. 
But language models are much larger than image models.

Today, LLMs are the way to go if you want to give your products some smarts. And the only way you can get to use them is to run them on a cloud provider.

That's why Apple had to sign deals with cloud providers to be able to use their models, like Google's Gemini for Siri.

To be able to differentiate themselves from the competition, they put a lot of effort on privacy preservation features, like the ability to run models on their own private cloud compute, and even on YOUR specific cloud instance.

# But Why Go Local?

So right now, Apple is dependent on the cloud providers to be able to use the largest models. That's not unique to Apple: they have been using Google search since 2002 and Google actually pays them roughly $20B/year in ad revenue share.

But you can be sure that Apple will work hard to reduce this dependency.
Running models on your own hardware has definitely some advantages:
- privacy
- speed
- energy efficiency

Privacy is a big selling point for Apple, and the best way to protect it is to be able to run models on your own hardware, and if possible on the device itself. You don't have to send relevant private data when everything is running locally.

In terms of speed, even if your device has less compute power, you avoid network issues, and model latency issues can be reduced because you are the only user of the model: today you can feel congestion at some points of the day, as services like ChatGPT or Cursor slow to a halt.
You don't have to send data, and even if 5G network performance is good, the globally available bandwidth is growing much slower than the compute power. And same thing with tasks the agent may want to perform: they will run locally too, so no back and forth between the device and the cloud.

And finally, nothing beats mobile devices for energy efficiency: in a time of energy shortage, running models on the device itself is the way to go.


# The Transition

So how will this transition happen, and when ?

It will take some time, and some significant efforts, but the stakes are as high as it gets.

The main issue is the size of the models. But some models are small enough to run on the device.
And techniques to compress models are getting better and better: quantization, pruning, distillation, mixture of experts, LORAs, fine-tuning, better training datasets are only some of the ways to do more with less.
The result is a rapid improvement in the ratio of model quality to model size: models like Llama 3.3 or Qwen 3 are approaching the quality of the largest models a year or two ago, while being small enough to run on the device.

So what kind of tasks would be candidates for running locally?
- Computational photography has been running locally for a while now.
- Speech recognition is there too, and it's already very good.
- For image generation, we are on the verge of being able to routinely run them on the device. Apple has already included it in Image Playground: even if it is still limited, everything is running on the device
- Assistants (like Siri) can use specific models for text extraction (like people, dates, places, etc.)
- And of course, LLMs: some models like Llama 3.3 or Qwen 3 we mentioned earlier are small enough to run on the device

So, we are already in the middle of the transition: AI is moving from the cloud to local devices, one model at a time.

Of course there will always be some room for giant cloud-based models, as there is no such thing as too much intelligence. But do you really need a Nobel-Prize-level intelligence to find you a nearby restaurant? Not really. So a cascade of models, from the one running in your watch to the one taking a full datacenter to run, is the best way to optimize the use of compute power.


<div class="callout-highlight">
And that means that, at some point, **local models will regain some share of the AI market**, and companies like Apple (with their Neural Engine) and Google (with their mobile-specific TPUs) will be there to benefit from it.</div>

# The future of the future

Next time, we will go even further in the future (and in the past!), and discover what's beyond the current paradigms of computing: in a word, let there be light!