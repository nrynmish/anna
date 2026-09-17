# ANNA Failure Analysis

This analysis uses the saved 200-example end-to-end golden evaluation. No additional model inference was performed.

## 1. Multi-issue messages are difficult to classify

### Evidence

Several messages contain multiple support problems, causing the classifier to select an intent associated with only one part of the message.

- **#10:** payment issue + request for ride pass → gold `payments_charges`, predicted `promotions_discounts`.
- **#12:** signup + driver cancellation + blocked account → gold `account_access`, predicted `signup_onboarding`.
- **#41:** card charged after driver cancellation → gold `refunds_adjustments`, predicted `driver_behavior`.
- **#98:** cancellation charge + rejection of offered credit → gold `refunds_adjustments`, predicted `payments_charges`.

### Hypothesis

The current intent stage assigns one primary label without explicitly decomposing a message into separate issues. Strong lexical evidence from a secondary issue can therefore override the operationally dominant intent.

### Next step

Introduce issue extraction before intent classification: split multi-issue messages into atomic support problems, classify each, then select the routing-relevant primary intent using an explicit precedence policy.

## 2. `unclear` and `other` absorb semantically understandable requests

### Evidence

The `unclear` class has precision 0.1750, recall 0.5385, and F1 0.2642, substantially weaker than most primary intents.

Examples include:

- **#5:** driver app is not adding up rides → predicted `unclear` instead of `payouts_and_earnings`.
- **#13:** asks for Uber Select vehicle models → predicted `other`.
- **#21:** explicitly says the account is locked and needs unlocking → predicted `other`.
- **#36:** newly joined driver cannot use the driver app → predicted `unclear`.

### Hypothesis

The deterministic labeler and downstream classifier are conservative when messages contain support-history noise, multiple clauses, or vocabulary not strongly represented in the taxonomy examples.

### Next step

Replace or augment the heuristic intent stage with a semantic few-shot classifier using representative examples from the golden set, while retaining `unclear` as a calibrated low-confidence state.

## 3. Account-specific requests create a capability boundary

### Evidence

ANNA escalates requests that require access to private account, trip, payment, or driver information even when the underlying intent is recognizable.

Examples include:

- **#9:** lost bag and request to contact a driver from a past trip.
- **#17:** request to change the account phone number.
- **#21:** request to unlock a locked account.
- **#90:** lost phone followed by difficulty contacting the driver.

### Hypothesis

The system is correctly detecting that historical support evidence cannot substitute for live account state. The current capability boundary therefore limits automation coverage even when retrieval is strong.

### Next step

Keep these capability gates for the current architecture. If live integrations are introduced later, expose narrowly scoped tools for account/trip verification rather than weakening the escalation policy.

## 4. Retrieval similarity alone is insufficient evidence

### Evidence

Some examples have strong semantic similarity but weak evidence agreement, while others have both strong similarity and strong evidence.

- **#58:** similarity 0.738, evidence 0.330.
- **#79:** similarity 0.744, evidence 0.329.
- **#105:** similarity 0.676, evidence 0.296.
- **#40:** similarity 0.956, evidence 0.838.

### Hypothesis

Semantically similar cases can have different resolutions. The historical corpus also contains many cases where support redirected the customer to another channel rather than resolving the issue directly. Treating similarity as equivalent to resolution precedent would therefore overstate grounding quality.

### Next step

Improve evidence scoring by distinguishing substantive resolutions from generic redirects, grouping evidence by resolution type, and requiring agreement among genuinely resolving precedents.

## 5. Conservative routing produces low automation coverage

### Evidence

Among 139 routing-labeled examples, 105 were labeled safe to auto-handle and 34 were labeled for escalation. ANNA auto-handled 21 cases and escalated 118. Safe auto-handle precision was 1.0000 with zero unsafe auto-handles, but 84 gold-auto cases were escalated.

### Hypothesis

The current policy is optimized toward avoiding unsafe automation. Multiple independent gates — intent confidence, retrieval similarity, evidence quality, generation confidence, financial-resolution agreement, and risk/capability checks — compound conservatism.

### Next step

Do not immediately lower thresholds. First improve intent confidence calibration and evidence quality, then re-evaluate threshold trade-offs on a larger routing benchmark. The target should be higher safe coverage without increasing unsafe auto-handles.

## Summary

The dominant failure pattern is not uncontrolled hallucination: the current system produced zero unsafe auto-handles on the routing-labeled golden set. The larger limitation is conservative handling and imperfect intent resolution on multi-issue, noisy, and underspecified messages. The next iteration should therefore focus on semantic intent classification, multi-issue decomposition, and better resolution-aware evidence scoring before relaxing routing thresholds.
