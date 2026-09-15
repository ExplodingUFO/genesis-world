"""Public checkpoint restoration must retain contact history and fail before mutation."""

import numpy as np
import pytest

import genesis as gs


def test_rigid_checkpoint_restores_contact_history_and_rejects_partial_state():
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.049)))
    scene.build(n_envs=1)
    try:
        scene.step()
        solver = scene.rigid_solver
        saved = scene.dump_ckpt_to_numpy()
        for field in (
            solver.constraint_solver.constraint_state.qacc_ws,
            solver.collider._collider_state.first_time,
            solver.collider._collider_state.contact_cache.normal,
        ):
            original = field.to_numpy().copy()
            changed = original + np.ones_like(original)
            if original.dtype == np.bool_:
                changed = np.logical_not(original)
            field.from_numpy(changed)
            solver.load_ckpt_from_numpy(saved)
            np.testing.assert_array_equal(field.to_numpy(), original)

        position_before = box.get_pos().clone()
        qpos_key = next(key for key in saved if key.endswith(".qpos"))
        missing_key = next(key for key in saved if key.endswith(".qacc_ws"))
        invalid = dict(saved)
        invalid[qpos_key] = saved[qpos_key] + 1.0
        del invalid[missing_key]
        with pytest.raises(ValueError):
            solver.load_ckpt_from_numpy(invalid)
        np.testing.assert_array_equal(box.get_pos().cpu().numpy(), position_before.cpu().numpy())
        np.testing.assert_array_equal(scene.dump_ckpt_to_numpy()[qpos_key], saved[qpos_key])
    finally:
        scene.destroy()
