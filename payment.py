import os

import stripe
from dotenv import load_dotenv

load_dotenv()

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
stripe.api_key = STRIPE_SECRET_KEY


def _require_stripe_key() -> None:
    if not STRIPE_SECRET_KEY:
        raise RuntimeError("STRIPE_SECRET_KEY environment variable is not set.")


def create_trip_checkout_sessions(
    trip_id: str,
    trip_destination: str,
    total_cost: float,
    members: list[dict],
    split: bool,
    success_url: str,
    cancel_url: str,
    currency: str = "usd",
) -> list[dict]:
    """Create Stripe Checkout Session(s) for paying a trip.

    Each session is a real, independently-payable hosted Stripe page — this
    is what lets a given traveler pay with their own card on their own
    device without needing an account or login on this app (which doesn't
    have user/auth/database support yet). The organizer (current user, i.e.
    the browser calling this) is expected to redirect straight into their
    own session's checkout_url; sessions for other members are returned so
    the organizer can copy/share those links manually (e.g. via message)
    until the app has real per-member accounts to deliver these
    automatically per user.

    Two modes, matching the frontend's "Pay Individual" / "Split Bill"
    toggle:
    - split=True  -> one session per member. The total is divided in integer
      cents and the leftover cents go to the earliest members, so the sessions
      always add up to total_cost exactly (amounts can differ by $0.01).
    - split=False -> one session only, for whichever member is flagged
      isCurrentUser (falls back to the first member), for the full
      total_cost — this is the "I'll cover it" path.

    Args:
        trip_id: Identifier for the trip (used only as Stripe metadata for
            reconciliation — this app has no trip database yet).
        trip_destination: Human-readable trip destination, used in the
            Stripe line-item name shown on the hosted checkout page.
        total_cost: Full trip cost.
        members: List of {"id": str, "name": str, "isCurrentUser": bool}.
        split: Whether to split evenly across all members (True) or charge
            just the current user the full amount (False).
        success_url: Where Stripe redirects on successful payment. The
            literal string "{CHECKOUT_SESSION_ID}" is appended as a query
            param automatically — do not include your own session_id param.
        cancel_url: Where Stripe redirects if the payer backs out.
        currency: ISO currency code, lowercase (default "usd").

    Returns:
        List of dicts: {member_id, member_name, is_current_user, amount,
        currency, checkout_url, session_id}.
    """
    _require_stripe_key()

    if not members:
        raise RuntimeError("At least one trip member is required to create a payment session.")

    if split:
        targets = members
        # Split in integer cents and hand the leftover cents to the first few
        # members, so the sessions sum to total_cost EXACTLY. The old
        # round(total_cost / len(members), 2) charged every member the same
        # rounded figure, which under- or over-collected by up to
        # len(members)/2 cents (e.g. $100 across 3 people billed $99.99).
        total_cents = int(round(total_cost * 100))
        base_cents, remainder = divmod(total_cents, len(members))
        amounts_cents = [
            base_cents + (1 if i < remainder else 0) for i in range(len(members))
        ]
    else:
        current = next((m for m in members if m.get("isCurrentUser")), members[0])
        targets = [current]
        amounts_cents = [int(round(total_cost * 100))]

    if any(cents <= 0 for cents in amounts_cents):
        raise RuntimeError(
            f"Computed a non-positive charge amount ({min(amounts_cents) / 100}) — "
            f"check total_cost/members.")

    sessions: list[dict] = []
    for member, amount_cents in zip(targets, amounts_cents):
        amount = amount_cents / 100
        label = (
            f"{trip_destination} trip — {member.get('name', 'Traveler')}'s share"
            if split
            else f"{trip_destination} trip — full payment"
        )
        session = stripe.checkout.Session.create(
            mode="payment",
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": currency,
                    "unit_amount": amount_cents,
                    "product_data": {"name": label},
                },
                "quantity": 1,
            }],
            metadata={
                "trip_id": trip_id,
                "member_id": str(member.get("id", "")),
                "member_name": member.get("name", ""),
                "split": str(split),
            },
            success_url=f"{success_url}{'&' if '?' in success_url else '?'}session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=cancel_url,
        )
        sessions.append({
            "member_id": member.get("id"),
            "member_name": member.get("name"),
            "is_current_user": bool(member.get("isCurrentUser", False)),
            "amount": amount,
            "currency": currency,
            "checkout_url": session.url,
            "session_id": session.id,
        })

    return sessions


def get_checkout_session_status(session_id: str) -> dict:
    """Look up whether a given Checkout Session has been paid.

    Polling this from the frontend after a redirect back from Stripe is a
    stand-in for a proper webhook handler (checkout.session.completed) —
    fine for local/dev use, but a real deployment should verify payment via
    a signed webhook rather than trusting a client-supplied session_id.
    """
    _require_stripe_key()
    session = stripe.checkout.Session.retrieve(session_id)
    # session.metadata is a StripeObject, not a plain dict — dict(...) on it
    # fails on current stripe-python ("StripeObject is not iterable or a
    # mapping"). Use .to_dict() instead.
    metadata = session.metadata.to_dict() if session.metadata else {}
    return {
        "session_id": session.id,
        "payment_status": session.payment_status,  # "paid" | "unpaid" | "no_payment_required"
        "metadata": metadata,
    }