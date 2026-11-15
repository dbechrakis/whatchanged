# Adtributor notes

The formulas below are those implemented in `src/whatchanged/attribute.py`. Each carries its place in Bhagwan, Mahajan, Vemuri and Reddy, "Adtributor: Revenue Debugging in Advertising Systems", NSDI 2014. They were read from the paper itself.

Notation. For an element j of a dimension, F_j is its value on the baseline side (the paper's forecast) and A_j its value on the current side (the paper's actual). F and A are the totals.

## Explanatory power, additive measure (section 3.2, equation 4)

EP_j = (A_j minus F_j) over (A minus F).

It can exceed 100% or be negative when an element moves against the total. Over all elements of a dimension it sums to 100%.

## Surprise (section 3.3, equations 5 to 7)

p_j = F_j over F and q_j = A_j over A. The surprise of element j is the Jensen-Shannon divergence term

S_j = 0.5 times (p_j log2(2 p_j over (p_j plus q_j)) plus q_j log2(2 q_j over (p_j plus q_j))).

The paper bounds the divergence between 0 and 1, which fixes the logarithm base at 2.

## Explanatory power, derived measure m1 over m2 (section 4.1, equation 8)

Raw EP_j = ((A_j(m1) minus F_j(m1)) times F(m2) minus (A_j(m2) minus F_j(m2)) times F(m1)) over (F(m2) times (F(m2) plus A_j(m2) minus F_j(m2))).

The raw values are normalised so that they sum to 100% over the elements of the dimension. This is the finite-difference partial derivative of the ratio. The paper's worked example is in `tests/test_attribute.py`: cost per click with revenue as m1 and clicks as m2 gives 125%, 106%, minus 131% and 0% for the four advertisers.

## Surprise, derived measure (section 4.2)

The sum of the surprise of m1 and the surprise of m2, which assumes the two are independent. The paper calls this an approximation.

## The algorithm (section 3.4, Figure 2)

Within one dimension, order the elements by surprise, highest first. Walk down the list. An element joins the candidate set if its EP is above TEEP (Occam's razor). The surprise of the set is the sum of its elements' surprise. When the EP of the set is above TEP, the set is complete and the walk stops. A dimension whose walk ends without reaching TEP gives no candidate. Candidate sets are ordered by surprise and the top three are returned. The paper uses TEP of 67% and TEEP of 10%.

## What whatchanged adds

- A third answer. A set with `spread_at` elements or more (default 3) counts as the change being spread over the dimension and is not returned. When no dimension has a smaller set, the answer is "no segment explains it" and no set is offered.
- A recursive search. The same step is applied inside each element of the top set, over the remaining dimensions, to a depth the user sets.
- A dimension with a single element is skipped, since it explains everything trivially.
- No change to explain. When the metric is equal on both sides, the answer says so.

The paper does not give these. They are the package's choices, and the benchmark reports where they fail.
