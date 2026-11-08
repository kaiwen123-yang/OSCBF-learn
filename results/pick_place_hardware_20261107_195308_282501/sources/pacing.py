"""Monotonic pacing for whole six-axis CPV groups, independent of the SDK."""
import math
import time


class GroupPacer:
    def __init__(self, period_s=.01, *, clock=time.monotonic, sleep=time.sleep):
        if not math.isfinite(period_s) or period_s <= 0:
            raise ValueError('CPV group period must be positive and finite')
        self.period_s, self.clock, self.sleep = period_s, clock, sleep
        self.last_start = self.last_end = None
        self.waited_s = 0.

    def wait(self):
        if self.last_start is None:
            return
        # Anchor to the actual previous start, including after a late send.
        # Never send multiple groups to catch up with a missed old deadline.
        due = self.last_start + self.period_s
        started = self.clock()
        while True:
            remaining = due - self.clock()
            if remaining <= 0:
                break
            self.sleep(remaining)  # SIGINT/SIGTERM can interrupt the caller.
        self.waited_s += self.clock() - started

    def begin(self, started, *, immediate=False):
        due = None if self.last_start is None else self.last_start + self.period_s
        row = {'pacing_period_s': self.period_s, 'pacing_bypassed': immediate,
               'scheduled_start_s': due, 'pacing_wait_s': self.waited_s,
               'deadline_lateness_s': None if due is None else max(0., started-due),
               'interval_from_previous_start_s': None if self.last_start is None else started-self.last_start,
               'gap_from_previous_end_s': None if self.last_end is None else started-self.last_end}
        self.last_start, self.waited_s = started, 0.
        return row

    def end(self, ended):
        self.last_end = ended
