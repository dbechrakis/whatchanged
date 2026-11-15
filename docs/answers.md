# The three kinds of answer

Every result has a `kind`. The package never blurs them.

## exact

`change` and `decompose`. This is arithmetic. For a ratio, the change is split into one mix factor per level of the dimensions you give and a rate factor, by averaging each factor's effect over every order in which the factors can be switched. The parts add up to the change, and the function raises an error if they do not. It says where a difference sits. It does not say why it arose.

## heuristic

`explain`. This is Adtributor, a search from one paper with thresholds. It looks at one dimension at a time, ranks elements by how much their share moved, and keeps the few that account for most of the change. It can name the wrong segment. It can also answer that no segment explains the change, which happens when the change is spread over every element.

## statistical

`noise`. A bootstrap over units gives each named segment an interval, and a segment whose interval contains zero is flagged as within sampling noise. With earlier period pairs it also ranks the current contribution among them. It checks sampling noise only. It does not check seasonality, data errors or changes in definitions.

## The summary

`summary()` writes at most five plain sentences. It describes where the change sits. It never says why.
