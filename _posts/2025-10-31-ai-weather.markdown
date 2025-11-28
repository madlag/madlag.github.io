---
layout: post
title: "AI Weather Today: Cloudy with Risks of API Outages"
date: 2025-10-31
description: "What outages reveal about AI provider priorities"
author: francois
tags: [AI, LLM, Infrastructure]
categories: []
image: assets/images/posts/ai-weather/ai_weather_cover.png
featured: true
---

The Claude outage this morning made barely a ripple in the news. Probably most people are used to it by now. And of course it's not specific to Anthropic: OpenAI had their own outages from time to time too.

But outages are just the worst manifestation of a bigger problem: the complete reliance of AI-based businesses on AI API providers, combined with the fact that these providers are not able to scale to meet the demand.

<div style="clear: both;"></div>
### Musical chairs: our provider switching journey

At Random Walk we have been using AI everyday for almost 2 years now, just like a lot of people, to develop [Mimir](https://apps.apple.com/fr/app/mimir-questions-and-answers/id6511234817?l=en-GB), an AI private tutor for children.
But as AI-assisted AI-agent developers (what a mouthful !), we see too what's going on behind the scenes.

We saw Anthropic becoming the best for conversational agents, because of the quality of the interactions, instructions following, and external functions calling.
And we saw Anthropic API becoming victim of its success, to the point it became almost unusable in terms of latency.

Then we switched to OpenAI because it's of course very good, and quite decent in terms of speed.

And finally, this week we switched to Gemini. In terms of API features it's not the best, as there are some nasty bugs or missing features we have to deal with, but wow, that thing is fast, and the conversation quality is really good, so we are ready to bite the bullet and go for it.

You can see in [a previous post](/llm-comparison/) a more detailed comparison of the different providers, with specific metrics and features for real-time conversational apps. What's funny is that the observations we did almost one year ago are roughly still valid today.

### The morning surprise

So this morning Claude was not working, and the Anthropic API was at least partially down as well.
I was a bit disappointed when I started my day that I would have to wait a few hours to get back to code with my favorite agent in Cursor, claude-4.5-sonnet.
But I was wrong: Cursor was humming along just fine with the Anthropic API in the background ingesting my refactor requests and all the usual stuff.

### What does this all mean?

The landscape is not that complicated right now.
To run an AI business, you need compute, and you need to get money from your customers.

It's quite obvious Anthropic is focusing on the business side of things, and OpenAI is focusing on the consumer side of things.

Code models are just crazy good. That's just unbelievable. As a developer, I live in a fantasy land right now.

I am using the Ultra plan of Cursor, with mostly Claude 4.5 in the background. $200 per month. That may look expensive, but it is actually cheap as hell. I would pay 5x more for this if I had to.
The amount of work I can do with it in a single day is just amazing.

So yes, getting money is not an issue if you serve a code LLM. And Anthropic knows it: the way sonnet-4.5 is up in Cursor when it is down everywhere else is just a clear sign.

### What's next

That gives you an idea of our life right now: just like farmers, we look at the Cloud and we pray for the weather to be good. And we plan to make sure that we can adapt to climate change in the long run!

Hopefully, this is the first post in a series of posts about the AI landscape:
- practical advice about how to use code agents, and how they can make you more productive, happy and good looking.
- the AI battleground: bottlenecks, trends, battles, and who's going to win.

So see you soon !