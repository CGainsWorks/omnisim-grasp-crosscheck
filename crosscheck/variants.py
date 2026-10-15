"""The worlds the policies are evaluated in.

``nominal`` is the world every demonstration comes from. The controlled change
is a single parameter of the box, chosen on the MuJoCo reference before any
OmniSim run (see ``docs/preregistration.md``) so the OmniSim result could not
steer the choice.
"""

from crosscheck import task

VARIANTS = {
    "nominal": {"box_mass": task.BOX_MASS, "box_friction": task.BOX_FRICTION},
}

# Candidate changes screened on MuJoCo. Only the one named in
# docs/preregistration.md is run in OmniSim.
CANDIDATES = {
    "friction_0.30": {"box_mass": task.BOX_MASS, "box_friction": 0.30},
    "friction_0.15": {"box_mass": task.BOX_MASS, "box_friction": 0.15},
    "friction_0.08": {"box_mass": task.BOX_MASS, "box_friction": 0.08},
    "friction_0.07": {"box_mass": task.BOX_MASS, "box_friction": 0.07},
    "friction_0.06": {"box_mass": task.BOX_MASS, "box_friction": 0.06},
    "mass_x4": {"box_mass": task.BOX_MASS * 4, "box_friction": task.BOX_FRICTION},
    "mass_x10": {"box_mass": task.BOX_MASS * 10, "box_friction": task.BOX_FRICTION},
}
VARIANTS.update(CANDIDATES)
