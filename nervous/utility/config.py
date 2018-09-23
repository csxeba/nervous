import os


class StressedNetConfig:

    def __init__(self,
                 synaptic_environmental_constraint=0.8,
                 group_environmental_constraint=0.6,
                 stress_factor=0.8,
                 save_folder=os.path.expanduser("~/.nervous/models/")):
        self._synaptic_environmental_constraint = synaptic_environmental_constraint
        self._group_environmental_constraint = group_environmental_constraint
        self._stress_factor = stress_factor
        self._save_folder = save_folder
        self._sanitize()

    def _sanitize(self):
        if 1. < self._group_environmental_constraint <= 0.:
            raise ValueError("Group environmental constraint has to be in the range [0. - 1.)")
        if 1. < self._synaptic_environmental_constraint <= 0.:
            raise ValueError("Synaptic environmental constraint has to be in the range [0. - 1.)")
        if 1. < self._stress_factor <= 0.:
            raise ValueError("Stress factor has to be in the range [0. - 1.)")
        if not os.path.exists(self._save_folder):
            os.makedirs(self._save_folder)

    @property
    def synaptic_environmental_constraint(self):
        return self._synaptic_environmental_constraint

    @synaptic_environmental_constraint.setter
    def synaptic_environmental_constraint(self, value):
        self._synaptic_environmental_constraint = value
        self._sanitize()

    @property
    def group_environmental_constraint(self):
        return self._group_environmental_constraint

    @group_environmental_constraint.setter
    def group_environmental_constraint(self, value):
        self._group_environmental_constraint = value
        self._sanitize()

    @property
    def stress_factor(self):
        return self._stress_factor

    @stress_factor.setter
    def stress_factor(self, value):
        self._stress_factor = value
        self._sanitize()

    @property
    def save_folder(self):
        return self._save_folder

    @save_folder.setter
    def save_folder(self, value):
        self._save_folder = value
        self._sanitize()

    def __getitem__(self, item):
        if item == "self":
            raise ValueError("Hahaha")
        return self.__dict__[item]
