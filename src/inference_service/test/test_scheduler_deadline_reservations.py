import threading
import time

import pytest

from inference_service.scheduler.deadline_reservations import DeadlineReservationTable


@pytest.mark.parametrize(
    "field", ["pipeline_id", "session_id", "binding_id", "binding_incarnation", "expected_boot_id"]
)
def test_close_reconciles_only_the_drained_binding(field):
    table = DeadlineReservationTable()
    owner = dict(
        pipeline_id="primary",
        session_id="session",
        binding_id="binding",
        binding_incarnation=1,
        expected_boot_id="boot",
    )
    reservation = table.try_reserve(**owner, hardware_resource_id="ascend:0", now_ns=0, deadline_ns=100)
    table.mark_unknown(reservation)
    wrong_owner = {**owner, field: 2 if field == "binding_incarnation" else "other"}
    table.reconcile_binding(**wrong_owner)
    assert table.try_reserve(pipeline_id="other", hardware_resource_id="ascend:0", now_ns=200, deadline_ns=300) is None
    table.reconcile_binding(**owner)
    assert (
        table.try_reserve(pipeline_id="other", hardware_resource_id="ascend:0", now_ns=200, deadline_ns=300) is not None
    )


def test_close_before_unknown_mark_cannot_reintroduce_resource_quarantine():
    table = DeadlineReservationTable()
    owner = dict(
        pipeline_id="primary",
        session_id="session",
        binding_id="binding",
        binding_incarnation=1,
        expected_boot_id="boot",
    )
    reservation = table.try_reserve(**owner, hardware_resource_id="ascend:0", now_ns=0, deadline_ns=100)
    assert reservation is not None
    table.reconcile_binding(**owner)
    table.mark_unknown(reservation)
    assert (
        table.try_reserve(pipeline_id="other", hardware_resource_id="ascend:0", now_ns=200, deadline_ns=300) is not None
    )


def test_same_resource_reservations_are_serialized() -> None:
    table = DeadlineReservationTable()
    first = table.try_reserve(
        pipeline_id="primary",
        hardware_resource_id="ascend:0",
        now_ns=100,
        deadline_ns=300,
        estimate_ns=80,
    )
    second = table.try_reserve(
        pipeline_id="fallback",
        hardware_resource_id="ascend:0",
        now_ns=110,
        deadline_ns=300,
        estimate_ns=90,
    )

    assert first is not None
    assert second is not None
    assert (first.estimated_start_ns, first.estimated_finish_ns) == (100, 180)
    assert (second.estimated_start_ns, second.estimated_finish_ns) == (180, 270)


def test_same_resource_dispatch_waits_for_predecessor_release() -> None:
    table = DeadlineReservationTable()
    now_ns = time.monotonic_ns()
    first = table.try_reserve(
        pipeline_id="primary",
        hardware_resource_id="ascend:0",
        now_ns=now_ns,
        deadline_ns=now_ns + 1_000_000_000,
        estimate_ns=20_000_000,
    )
    second = table.try_reserve(
        pipeline_id="fallback",
        hardware_resource_id="ascend:0",
        now_ns=now_ns,
        deadline_ns=now_ns + 1_000_000_000,
        estimate_ns=20_000_000,
    )
    assert first is not None and second is not None
    completed = threading.Event()
    result: list[str] = []

    waiter = threading.Thread(
        target=lambda: (result.append(table.wait_for_turn(second, deadline_ns=now_ns + 1_000_000_000)), completed.set())
    )
    waiter.start()
    assert not completed.wait(0.05)
    table.release(first)
    assert completed.wait(1.0)
    waiter.join(timeout=1.0)
    assert result == ["ready"]
    table.release(second)


def test_dispatch_turn_rechecks_deadline_against_actual_start() -> None:
    table = DeadlineReservationTable()
    first = table.try_reserve(
        pipeline_id="primary",
        hardware_resource_id="ascend:0",
        now_ns=100,
        deadline_ns=1_000,
        estimate_ns=100,
    )
    second = table.try_reserve(
        pipeline_id="fallback",
        hardware_resource_id="ascend:0",
        now_ns=100,
        deadline_ns=350,
        estimate_ns=100,
    )
    assert first is not None and second is not None
    table.release(first)

    assert table.wait_for_turn(second, deadline_ns=350, now_ns=lambda: 300) == "deadline_exceeded"


def test_reservation_rejects_when_existing_work_exhausts_deadline() -> None:
    table = DeadlineReservationTable()
    assert table.try_reserve(
        pipeline_id="first",
        hardware_resource_id="ascend:0",
        now_ns=100,
        deadline_ns=300,
        estimate_ns=150,
    )
    assert (
        table.try_reserve(
            pipeline_id="second",
            hardware_resource_id="ascend:0",
            now_ns=110,
            deadline_ns=300,
            estimate_ns=60,
        )
        is None
    )


def test_different_resources_do_not_share_a_limit() -> None:
    table = DeadlineReservationTable()
    first = table.try_reserve(
        pipeline_id="first",
        hardware_resource_id="ascend:0",
        now_ns=100,
        deadline_ns=200,
        estimate_ns=100,
    )
    second = table.try_reserve(
        pipeline_id="second",
        hardware_resource_id="ascend:1",
        now_ns=100,
        deadline_ns=200,
        estimate_ns=100,
    )

    assert first is not None
    assert second is not None
    assert second.estimated_start_ns == 100


def test_not_started_and_completed_work_release_reservations() -> None:
    table = DeadlineReservationTable()
    reservation = table.try_reserve(
        pipeline_id="primary",
        hardware_resource_id="ascend:0",
        now_ns=100,
        deadline_ns=200,
        estimate_ns=100,
    )
    assert reservation is not None

    table.release(reservation)

    replacement = table.try_reserve(
        pipeline_id="fallback",
        hardware_resource_id="ascend:0",
        now_ns=110,
        deadline_ns=210,
        estimate_ns=100,
    )
    assert replacement is not None
    assert replacement.estimated_start_ns == 110


def test_unknown_quarantines_resource_until_pipeline_reboot() -> None:
    table = DeadlineReservationTable()
    reservation = table.try_reserve(
        pipeline_id="primary",
        hardware_resource_id="ascend:0",
        now_ns=100,
        deadline_ns=200,
        estimate_ns=100,
    )
    assert reservation is not None
    table.mark_unknown(reservation)

    assert (
        table.try_reserve(
            pipeline_id="fallback",
            hardware_resource_id="ascend:0",
            now_ns=300,
            deadline_ns=500,
            estimate_ns=100,
        )
        is None
    )

    table.reconcile_pipeline("primary")
    assert table.try_reserve(
        pipeline_id="fallback",
        hardware_resource_id="ascend:0",
        now_ns=300,
        deadline_ns=500,
        estimate_ns=100,
    )


def test_edf_orders_waiting_work_by_deadline_with_fifo_ties() -> None:
    table = DeadlineReservationTable(policy="edf")
    blocker = table.try_reserve(pipeline_id="active", hardware_resource_id="ascend:0", now_ns=100, deadline_ns=1_000)
    assert blocker is not None
    assert table.wait_for_turn(blocker, deadline_ns=1_000, now_ns=lambda: 105) == "ready"
    later = table.try_reserve(pipeline_id="later", hardware_resource_id="ascend:0", now_ns=110, deadline_ns=900)
    earliest = table.try_reserve(pipeline_id="earliest", hardware_resource_id="ascend:0", now_ns=120, deadline_ns=700)
    tied = table.try_reserve(pipeline_id="tied", hardware_resource_id="ascend:0", now_ns=130, deadline_ns=700)
    assert later is not None and earliest is not None and tied is not None
    table.release(blocker)
    assert table.wait_for_turn(earliest, deadline_ns=700, now_ns=lambda: 150) == "ready"
    table.release(earliest)
    assert table.wait_for_turn(tied, deadline_ns=700, now_ns=lambda: 160) == "ready"
    table.release(tied)
    assert table.wait_for_turn(later, deadline_ns=900, now_ns=lambda: 170) == "ready"


def test_edf_reorders_reservations_that_have_not_started() -> None:
    table = DeadlineReservationTable(policy="edf")
    later = table.try_reserve(
        pipeline_id="later",
        hardware_resource_id="ascend:0",
        now_ns=100,
        deadline_ns=900,
    )
    earlier = table.try_reserve(
        pipeline_id="earlier",
        hardware_resource_id="ascend:0",
        now_ns=110,
        deadline_ns=700,
    )
    assert later is not None and earlier is not None

    assert table.wait_for_turn(earlier, deadline_ns=700, now_ns=lambda: 120) == "ready"
    table.release(earlier)
    assert table.wait_for_turn(later, deadline_ns=900, now_ns=lambda: 130) == "ready"


def test_edf_does_not_preempt_an_active_reservation() -> None:
    table = DeadlineReservationTable(policy="edf")
    active = table.try_reserve(pipeline_id="active", hardware_resource_id="ascend:0", now_ns=100, deadline_ns=1_000)
    assert active is not None
    assert table.wait_for_turn(active, deadline_ns=1_000, now_ns=lambda: 110) == "ready"
    urgent = table.try_reserve(
        pipeline_id="urgent",
        hardware_resource_id="ascend:0",
        now_ns=time.monotonic_ns(),
        deadline_ns=time.monotonic_ns() + 1_000_000_000,
    )
    assert urgent is not None

    completed = threading.Event()
    result: list[str] = []
    waiter = threading.Thread(
        target=lambda: (result.append(table.wait_for_turn(urgent, deadline_ns=urgent.deadline_ns)), completed.set())
    )
    waiter.start()
    assert not completed.wait(0.05)
    table.release(active)
    assert completed.wait(1.0)
    waiter.join(timeout=1.0)
    assert result == ["ready"]


def test_edf_expires_queued_work_without_a_profile_estimate() -> None:
    table = DeadlineReservationTable(policy="edf")
    reservation = table.try_reserve(pipeline_id="policy", hardware_resource_id="ascend:0", now_ns=100, deadline_ns=200)
    assert reservation is not None
    assert reservation.estimate_ns is None
    assert table.wait_for_turn(reservation, deadline_ns=200, now_ns=lambda: 200) == "deadline_exceeded"


def test_edf_waiter_discards_an_expired_predecessor() -> None:
    table = DeadlineReservationTable(policy="edf")
    expired = table.try_reserve(
        pipeline_id="expired",
        hardware_resource_id="ascend:0",
        now_ns=100,
        deadline_ns=150,
    )
    ready = table.try_reserve(
        pipeline_id="ready",
        hardware_resource_id="ascend:0",
        now_ns=110,
        deadline_ns=300,
    )
    assert expired is not None and ready is not None

    assert table.wait_for_turn(ready, deadline_ns=300, now_ns=lambda: 160) == "ready"
    assert table.wait_for_turn(expired, deadline_ns=150, now_ns=lambda: 160) == "reservation_released"


def test_edf_rejects_profile_finish_admission() -> None:
    table = DeadlineReservationTable(policy="edf")
    with pytest.raises(ValueError, match="does not accept"):
        table.try_reserve(
            pipeline_id="policy",
            hardware_resource_id="ascend:0",
            now_ns=100,
            deadline_ns=200,
            estimate_ns=50,
        )


def test_edf_concurrent_waiters_dispatch_in_deadline_order() -> None:
    table = DeadlineReservationTable(policy="edf")
    now_ns = time.monotonic_ns()
    blocker = table.try_reserve(
        pipeline_id="active",
        hardware_resource_id="ascend:0",
        now_ns=now_ns,
        deadline_ns=now_ns + 2_000_000_000,
    )
    assert blocker is not None
    assert table.wait_for_turn(blocker, deadline_ns=blocker.deadline_ns) == "ready"
    later = table.try_reserve(
        pipeline_id="later",
        hardware_resource_id="ascend:0",
        now_ns=now_ns,
        deadline_ns=now_ns + 1_500_000_000,
    )
    earlier = table.try_reserve(
        pipeline_id="earlier",
        hardware_resource_id="ascend:0",
        now_ns=now_ns,
        deadline_ns=now_ns + 1_000_000_000,
    )
    assert later is not None and earlier is not None
    order: list[str] = []

    def wait(name, reservation):
        assert table.wait_for_turn(reservation, deadline_ns=reservation.deadline_ns) == "ready"
        order.append(name)
        table.release(reservation)

    later_thread = threading.Thread(target=wait, args=("later", later))
    earlier_thread = threading.Thread(target=wait, args=("earlier", earlier))
    later_thread.start()
    earlier_thread.start()
    time.sleep(0.05)
    table.release(blocker)
    later_thread.join(timeout=1.0)
    earlier_thread.join(timeout=1.0)

    assert order == ["earlier", "later"]
