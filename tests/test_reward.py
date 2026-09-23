import cv2
import numpy as np

from osuai.actions import ActionSpace
from osuai.config import DetectorConfig, RewardConfig
from osuai.reward import (CircleDetector, CreditAssigner, GameCounters, HitTracker, PendingStep,
                          RewardShaper, teacher_action)
from osuai.tosu import parse_message


def _frame(ring_scale):
    img = np.zeros((192, 256, 3), np.uint8)
    cv2.circle(img, (128, 96), 16, (255, 128, 64), -1)
    cv2.circle(img, (128, 96), 16, (255, 255, 255), 2)
    if ring_scale:
        cv2.circle(img, (128, 96), int(16 * ring_scale), (255, 255, 255), 2)
    return img


def test_detector_urgency_grows_as_ring_shrinks():
    det = CircleDetector(DetectorConfig())
    far = det.detect(_frame(3.5))
    near = det.detect(_frame(1.6))
    assert len(far) == 1 and len(near) == 1
    assert abs(far[0].x - 0.5) < 0.02 and abs(far[0].y - 0.5) < 0.02
    assert near[0].urgency > far[0].urgency
    assert far[0].urgency < 0.4


def _closing_dets():
    """Кольцо почти сомкнулось на прошлом кадре, на текущем слилось с кругом → пора жать."""
    det = CircleDetector(DetectorConfig())
    det.detect(_frame(1.3))
    return det.detect(_frame(0))


def test_detector_merged_ring_is_urgent_only_after_closing():
    assert _closing_dets()[0].urgency == 1.0
    fresh = CircleDetector(DetectorConfig()).detect(_frame(0))  # новый объект без кольца
    assert fresh[0].urgency < 0.9


def test_detector_splits_overlapping_circles():
    img = np.zeros((192, 256, 3), np.uint8)
    for x in (100, 124):  # «стрим»: круги перекрываются
        cv2.circle(img, (x, 96), 16, (64, 200, 255), -1)
        cv2.circle(img, (x, 96), 16, (255, 255, 255), 2)
    dets = CircleDetector(DetectorConfig()).detect(img)
    assert len(dets) == 2
    assert sorted(round(d.x * 256) for d in dets) == [100, 124]


def test_detector_ignores_dark_frame():
    assert CircleDetector(DetectorConfig()).detect(np.zeros((192, 256, 3), np.uint8)) == []


def test_shaper_rewards_good_click_and_penalizes_spam():
    dets = _closing_dets()
    sh = RewardShaper(RewardConfig(), aspect=0.75)
    good = sh(dets, 0.5, 0.5, onset=True, scale=1.0)
    spam = sh(dets, 0.05, 0.05, onset=True, scale=1.0)
    idle = sh(dets, 0.5, 0.5, onset=False, scale=1.0)
    assert good > idle > 0 > spam


def test_teacher_points_at_circle():
    dets = _closing_dets()
    sp = ActionSpace(24, 32)
    a = teacher_action(dets, sp, prev_press=0)
    x, y, press = sp.to_norm(a)
    assert abs(x - 0.5) < 0.05 and abs(y - 0.5) < 0.05 and press == 1
    assert sp.to_norm(teacher_action(dets, sp, prev_press=1))[2] == 0


def test_credit_assigner_attributes_hit_to_onset():
    sink = []
    ca = CreditAssigner(RewardConfig(credit_window_steps=5, miss_lag_steps=1), lambda *a: sink.append(a))
    for i in range(4):
        ca.push(PendingStep(np.zeros(1), i, 0.0, i == 0, onset=(i == 1)))
    ca.apply_events(GameCounters(h300=1, combo=1))
    ca.apply_events(GameCounters(miss=1))
    ca.flush()
    rewards = [s[2] for s in sink]
    assert rewards[1] == 1.0          # попадание — шагу с нажатием
    assert rewards[2] == -1.0         # промах — с лагом 1 от последнего шага
    assert sink[-1][4] is True        # последний шаг терминальный
    assert sink[0][3] is True


def test_hit_tracker_handles_restart():
    ht = HitTracker()
    assert ht.delta(GameCounters(h300=5)) == GameCounters()
    assert ht.delta(GameCounters(h300=7, miss=1)).h300 == 2
    assert ht.delta(GameCounters(h300=0)) == GameCounters()  # рестарт карты


def test_parse_gosumemory_message():
    msg = {"menu": {"state": 2, "bm": {"time": {"current": 1234}}},
           "gameplay": {"score": 1000, "accuracy": 98.5, "combo": {"current": 12, "max": 30},
                        "hp": {"normal": 150.0},
                        "hits": {"300": 10, "100": 2, "50": 1, "0": 3, "sliderBreaks": 1}}}
    st = parse_message(msg, 0.0)
    assert st.playing and st.time_ms == 1234
    assert st.counters == GameCounters(10, 2, 1, 3, 1, 12)
    assert parse_message({}, 0.0).state == -1
