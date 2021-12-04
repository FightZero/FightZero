# this file defines available actions
# reference: https://github.com/TeamFightingICE/FightingICE/blob/master/python/Feature%20Extractor%20in%20Python/action.py

class Actions:
    def __init__(self):
        # map digits to actions
        self.actions = [
            "NEUTRAL",
            "STAND",
            "FORWARD_WALK",
            "DASH",
            "BACK_STEP",
            "CROUCH",
            "JUMP",
            "FOR_JUMP",
            "BACK_JUMP",
            "AIR",
            "STAND_GUARD",
            "CROUCH_GUARD",
            "AIR_GUARD",
            "STAND_GUARD_RECOV",
            "CROUCH_GUARD_RECOV",
            "AIR_GUARD_RECOV",
            "STAND_RECOV",
            "CROUCH_RECOV",
            "AIR_RECOV",
            "CHANGE_DOWN",
            "DOWN",
            "RISE",
            "LANDING",
            "THROW_A",
            "THROW_B",
            "THROW_HIT",
            "THROW_SUFFER",
            "STAND_A",
            "STAND_B",
            "CROUCH_A",
            "CROUCH_B",
            "AIR_A",
            "AIR_B",
            "AIR_DA",
            "AIR_DB",
            "STAND_FA",
            "STAND_FB",
            "CROUCH_FA",
            "CROUCH_FB",
            "AIR_FA",
            "AIR_FB",
            "AIR_UA",
            "AIR_UB",
            "STAND_D_DF_FA",
            "STAND_D_DF_FB",
            "STAND_F_D_DFA",
            "STAND_F_D_DFB",
            "STAND_D_DB_BA",
            "STAND_D_DB_BB",
            "AIR_D_DF_FA",
            "AIR_D_DF_FB",
            "AIR_F_D_DFA",
            "AIR_F_D_DFB",
            "AIR_D_DB_BA",
            "AIR_D_DB_BB",
            "STAND_D_DF_FC"
        ]
        # map action to digits
        self.actions_map = {
            m:i for i, m in enumerate(self.actions)
        }
        # set number of actions
        self.count = 56
        assert len(self.actions) == self.count
        # set other types of actions
        self.actions_digits_useless = [
            self.actions_map[m] for m in [
                "STAND",
                "AIR",
                "STAND_GUARD_RECOV",
                "CROUCH_GUARD_RECOV",
                "AIR_GUARD_RECOV",
                "STAND_RECOV",
                "CROUCH_RECOV",
                "AIR_RECOV",
                "CHANGE_DOWN",
                "DOWN",
                "RISE",
                "LANDING",
                "THROW_HIT",
                "THROW_SUFFER"
            ]
        ]
        self.actions_digits_useful = [
            i for i in range(self.count) if i not in self.actions_digits_useless
        ]
        self.count_useful = len(self.actions_digits_useful)
        self.actions_map_useful = {
            i:self.actions[n] for i,n in enumerate(self.actions_digits_useful)
        }
        self.actions_digits_air = [
            self.actions_map[m] for m in [
                "AIR_GUARD",
                "AIR_A",
                "AIR_B",
                "AIR_DA",
                "AIR_DB",
                "AIR_FA",
                "AIR_FB",
                "AIR_UA",
                "AIR_UB",
                "AIR_D_DF_FA",
                "AIR_D_DF_FB",
                "AIR_F_D_DFA",
                "AIR_F_D_DFB",
                "AIR_D_DB_BA",
                "AIR_D_DB_BB"
            ]
        ]
        self.actions_digits_ground = [
            self.actions_map[m] for m in [
                "STAND_D_DB_BA",
                "BACK_STEP",
                "FORWARD_WALK",
                "DASH",
                "JUMP",
                "FOR_JUMP",
                "BACK_JUMP",
                "STAND_GUARD",
                "CROUCH_GUARD",
                "THROW_A",
                "THROW_B",
                "STAND_A",
                "STAND_B",
                "CROUCH_A",
                "CROUCH_B",
                "STAND_FA",
                "STAND_FB",
                "CROUCH_FA",
                "CROUCH_FB",
                "STAND_D_DF_FA",
                "STAND_D_DF_FB",
                "STAND_F_D_DFA",
                "STAND_F_D_DFB",
                "STAND_D_DB_BB"
            ]
        ]