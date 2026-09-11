"""Uber-specific ANNA intent taxonomy (static, versioned).

This file defines a deterministic, versioned taxonomy derived from
the Uber_Support reconstruction and analysis in /artifacts.

Fields for each intent:
 - id: stable machine-readable id
 - name: human readable name
 - definition: concise definition of the intent
 - inclusion_criteria: what to include
 - exclusion_criteria: what to exclude (to disambiguate)
 - examples: 2-3 representative example utterances from the dataset

TAXONOMY is intentionally static and should be treated as read-only.
"""

TAXONOMY_VERSION = "uber_v1"

# Primary intents (11) plus 'other' and 'unclear'
TAXONOMY = [
    {
        "id": "account_access",
        "name": "Account / Access",
        "definition": "Problems accessing or managing a user account, identity, or profile details.",
        "inclusion_criteria": "Login failures, password resets, email/phone on account, account closures, profile updates, receipt access.",
        "exclusion_criteria": "Payment disputes, trip complaints, driver issues, or requests that are purely informational about features.",
        "examples": [
            "your web account management is super broken. Can’t get my receipts. Keeps telling me to verify number and add cards.",
            "I am unable to reset pwd of my login.not getting any email or support from u"
        ],
    },

    {
        "id": "payments_charges",
        "name": "Payments / Charges",
        "definition": "Customer questions or disputes about charges, payment methods, card errors, or unexpected charges.",
        "inclusion_criteria": "Reports of being charged, card/credit issues, wallet/balance visibility, unexpected fees or charge questions.",
        "exclusion_criteria": "Requests for refunds (use refunds_and_adjustments), payout questions for drivers (payouts_and_earnings), or promotional credits (promotions_discounts).",
        "examples": [
            "was charged for a trip I didn't take",
            "cancellation fee was applied to my account"
        ],
    },

    {
        "id": "refunds_adjustments",
        "name": "Refunds / Fare Adjustments",
        "definition": "Requests for refunds, fare corrections, cancellation fee waivers, or reimbursements.",
        "inclusion_criteria": "Explicit refund requests, fare disputes, chargebacks, or asking for money to be returned or corrected.",
        "exclusion_criteria": "General payment troubleshooting (payments_charges) or driver pay issues (payouts_and_earnings).",
        "examples": [
            "my driver broke down any tips on getting a refund",
            "I accidentally started trip but rider never got in the car & I don't know how to get the charge refunded"
        ],
    },

    {
        "id": "rides_and_trips",
        "name": "Rides / Trip Experience",
        "definition": "Issues or feedback about an individual trip: no-show, missing pickup, scheduling, or service experience.",
        "inclusion_criteria": "Problems with a specific ride, scheduled/airport pickups, cancellations, arrival/delay problems, and general trip complaints.",
        "exclusion_criteria": "Driver conduct complaints (driver_behavior), lost items (lost_and_found), or payment disputes (payments_charges).",
        "examples": [
            "I had booked uber for airport drop last night. You send a regret msg in the morning, well after the pickup window",
            "This scheduled airport drop disappears. SICKENING!"
        ],
    },

    {
        "id": "driver_behavior",
        "name": "Driver-related Issues",
        "definition": "Complaints or reports about driver conduct, cancellations, rude behavior, or driver-specific service problems.",
        "inclusion_criteria": "Rude or unsafe driver reports, driver cancellations, requests to report drivers, and driver performance complaints.",
        "exclusion_criteria": "Requests about contacting a driver about a lost item (lost_and_found) or driver payments (payouts_and_earnings).",
        "examples": [
            "how can i report the rude uber driver that cancelled the trip of my friend. Thank you...",
            "my driver cancelled the trip of my friend"
        ],
    },

    {
        "id": "lost_and_found",
        "name": "Lost Items / Returning Belongings",
        "definition": "Customer requests and help to recover items left in a vehicle or to contact a past driver about a lost belonging.",
        "inclusion_criteria": "Messages asking how to contact a driver about a lost item, reporting items left in a trip, and lost-and-found workflows.",
        "exclusion_criteria": "Trip complaints unrelated to an item, or driver conduct reports (driver_behavior).",
        "examples": [
            "I just left my bag in an Uber. How do you contact the driver of a past trip?",
            "It doesn't work. I got this message again \"We're sorry.Something went wrong. Please try again in a moment\""
        ],
    },

    {
        "id": "promotions_discounts",
        "name": "Promotions / Discounts / Credits",
        "definition": "Requests about promo codes, ride passes, credits, or how promotional balances are applied.",
        "inclusion_criteria": "Promo code problems, questions about ride passes, promotional credit application, and expirations.",
        "exclusion_criteria": "Refund requests for promotions (refunds_adjustments) or wallet/balance charge issues (payments_charges).",
        "examples": [
            "hey how can I get the ride pass for ride share?",
            "promo code isn't applying when I try to use it"
        ],
    },

    {
        "id": "app_technical",
        "name": "App / Technical Issues",
        "definition": "Bugs, errors, and technical problems with the mobile app, website, or in-app features.",
        "inclusion_criteria": "App crashes, features not working, errors in the app, API/link failures, and general technical outages.",
        "exclusion_criteria": "Account authentication problems (account_access) or trip-specific service complaints (rides_and_trips).",
        "examples": [
            "my app isnt adding up my rides and hasnt all day. is that resolved?",
            "I'm not getting your texts / why is there no phone number i'm literally stranded"
        ],
    },

    {
        "id": "signup_onboarding",
        "name": "Sign-up / Onboarding",
        "definition": "Questions about creating accounts, driver sign-up, background checks, or onboarding steps.",
        "inclusion_criteria": "Application status, background check questions, how to sign-up to be a driver or rider, and verification steps.",
        "exclusion_criteria": "Account access problems after onboarding (account_access) or driver conduct complaints (driver_behavior).",
        "examples": [
            "I've been applying with Uber for 1 1/2 month to be a driver. I haven't any updates yet!",
            "how can I sign-up my account"
        ],
    },

    {
        "id": "payouts_and_earnings",
        "name": "Payouts / Driver Earnings",
        "definition": "Driver-facing questions about earnings, cashouts, missing payouts, or pay/account payout issues.",
        "inclusion_criteria": "Cashout failures, missing earnings, payout processing problems, and driver balance questions.",
        "exclusion_criteria": "Rider refunds or wallet credits (refunds_adjustments / promotions_discounts) or account access issues (account_access).",
        "examples": [
            "Our team is working as quickly as possible to resolve this issue so that you may be able to cash out your earnings.",
            "Hi, am still unable to login. Showing Bad Request. How long it will take."
        ],
    },

    {
        "id": "safety_incident",
        "name": "Safety / Incident Reports",
        "definition": "Reports of safety events, personal injury, assault, or other incidents requiring escalation or investigation.",
        "inclusion_criteria": "Allegations of assault, injury on the platform, severe misconduct, or requests for urgent escalation.",
        "exclusion_criteria": "General complaints about driver behavior that do not allege harm (driver_behavior) or policy questions.",
        "examples": [
            "when i was attacked while on the job you ignored me you owe me money!!",
            "I didn't even get confronted on the situation. I had to find out I was pulled from the road when I spent my last dollar"
        ],
    },

    # catch-all categories
    {
        "id": "other",
        "name": "Other",
        "definition": "Catch-all for operationally meaningful support topics seen in the data but not captured by primary intents.",
        "inclusion_criteria": "Rare or brand-specific requests that are supported by the data but don't fit primary intents.",
        "exclusion_criteria": "Not for unclear/ambiguous messages; use `unclear` when the intent cannot be determined.",
        "examples": [
            "Please follow the attached link to send us a DM.",
        ],
    },

    {
        "id": "unclear",
        "name": "Unclear",
        "definition": "Messages where the customer's intent is ambiguous or cannot be determined from the text alone.",
        "inclusion_criteria": "Single-word replies, jokes, or messages lacking clear request/issue context.",
        "exclusion_criteria": "Do not use if the message matches any primary intent or `other` clearly.",
        "examples": [
            "Hollering 😂😂😂😂😂",
            "IM DEAD https://t.co/9ib0r41Ket"
        ],
    },
]


def get_intent_by_id(intent_id):
    for it in TAXONOMY:
        if it["id"] == intent_id:
            return it
    return None
