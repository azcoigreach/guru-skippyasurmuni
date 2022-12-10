# guru-skippyasurmuni
freqtrade.io strategy

Guru Skippyasurmuni - Trust the Awesomeness!!

Skippy is the name of my Crypto trading bot.  He is named after a charecter in the book series Expeditinary Force.  Where Skippy the Magnificent is a super intelligent ancient AI with a serious
additude problem.

Like Skippy, my Crypto bot likes to laugh at me when I think I know better than the algorythm. He ridicules me when I do not trust the awesomeness.  And frequently reminds my that I am an ingorant meat-bag.

With the added feature of `adjust_trade_position()`, Skippy has taken on a new persona.  He is now
Guru Skippyasurmuni.  The grand purveyor and seeker of all the muni.

Skippy's trading strategy is simple. He relies strictly on `buy` and `sell` positions based on
RSI and Aroon Oscillator indicators to populate signals. Coupled with a high count dollar-cost-averaging function to gradually and exponentially reposition the stake.

This version of Skippy's sub-mind relies heavily on the freqtrade strategy codebase GodStra written by [MaBlue on github](https://github.com/mablue).  Skippy's god-like powers come from the god-genes that MaBlue created.  Allowing Skippy to be easily hyper-opted (* certain conditions apply).  

Some help from PerkMeister on the [freqtrade.io Discord](https://discord.gg/kHaGH3wzHy)

## Installation

```git clone https://github.com/azcoigreach/guru-skippyasurmuni.git```

```cd guru-skippyasurmuni```

Edit the `user_data/guru-skippyasurmuni_config.json` and `user_data/strategies/GuruSkippyasurmuni_strategy.py` files to your needs. 

## Usage

```docker-compose up -d --build```

```docker-compose logs -f```