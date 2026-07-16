"""Checks for the immutable historical N9 evidence boundary."""

from cartpole_race.release import audit_authority_bytes, audit_banked_gates


def test_frozen_inputs_and_banked_rows_are_internally_consistent() -> None:
    authorities = audit_authority_bytes()
    gates = audit_banked_gates()
    assert len(authorities) == 13
    assert gates["total_successes"] == 72
    assert gates["total_trials"] == 72
    assert [record["seed"] for record in gates["files"]] == [12345, 777, 2024]
