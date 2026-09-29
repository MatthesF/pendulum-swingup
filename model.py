"""An N-link pendulum hanging from a cart that a motor pushes along a rail."""

import mujoco

LINK_LENGTH = 0.4
LINK_MASS = 0.1
CART_MASS = 0.5
CART_LIMIT = 2.4
MAX_CART_FORCE = 10.0
GRAVITY = 9.81
PHYSICS_TIMESTEP = 0.002


def build_model(links):
    spec = mujoco.MjSpec()
    spec.option.timestep = PHYSICS_TIMESTEP
    spec.option.gravity = [0, 0, -GRAVITY]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_RK4

    height = links * LINK_LENGTH
    distance = max(5.0, 2.5 * height)
    spec.worldbody.add_camera(
        name="overview", pos=[0, -distance, height / 2], xyaxes=[1, 0, 0, 0, 0, 1]
    )
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[-CART_LIMIT, 0, 0, CART_LIMIT, 0, 0],
        size=[0.015, 0, 0],
        contype=0,
        conaffinity=0,
    )

    cart = spec.worldbody.add_body(name="cart")
    cart.add_joint(
        name="rail",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[1, 0, 0],
        limited=True,
        range=[-CART_LIMIT, CART_LIMIT],
    )
    cart.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.12, 0.08, 0.06],
        mass=CART_MASS,
        contype=0,
        conaffinity=0,
        rgba=[0.2, 0.45, 0.8, 1],
    )

    # Each hinge sits at the tip of the link below and measures the angle against it;
    # zero is straight up.
    parent, pivot = cart, 0.0
    for _ in range(links):
        parent = parent.add_body(pos=[0, 0, pivot])
        parent.add_joint(type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
        parent.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0, 0, 0, 0, 0, LINK_LENGTH],
            size=[0.025, 0, 0],
            mass=LINK_MASS,
            contype=0,
            conaffinity=0,
            rgba=[0.9, 0.45, 0.15, 1],
        )
        pivot = LINK_LENGTH

    spec.add_actuator(
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        target="rail",
        gear=[1, 0, 0, 0, 0, 0],
        ctrllimited=True,
        ctrlrange=[-MAX_CART_FORCE, MAX_CART_FORCE],
    )
    return spec.compile()
