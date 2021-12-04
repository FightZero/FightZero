# implementation of PPO AI
import os
from typing import List, Tuple
from torch.utils.tensorboard.writer import SummaryWriter
from .Abstract import AIInterface
from ..Algorithms.PPO import PPO
from ..Utils.Actions import Actions

class PPOAI(AIInterface):
    def __init__(self, gateway, train=False):
        self.gateway = gateway
        # set whether in training mode
        self.training = train
        self.training_steps = 1e3
        self.training_steps_count = 0
        # set parameters
        self.actions = Actions()
        self.state_dimensions = 14 # NOTE: set correct number of dimensions here
        self.action_dimensions = self.actions.count_useful
        self.lr_actor = 1e-4
        self.lr_critic = 1e-3
        self.train_epochs = 80
        self.discount = 0.99
        self.eps_clip = 0.2
        self.reward_sum = 0
        self.sim_count = 0
        self.initialized = False
        self.states_map = {
            self.gateway.jvm.enumerate.State.AIR    : 0,
            self.gateway.jvm.enumerate.State.CROUCH : 1,
            self.gateway.jvm.enumerate.State.DOWN   : 2,
            self.gateway.jvm.enumerate.State.STAND  : 3,
        }
        self.hp_me, self.hp_opp = 400, 400 # NOTE: default 400 each side
        # create model
        self.model = PPO(
            self.state_dimensions,
            self.action_dimensions,
            self.lr_actor,
            self.lr_critic,
            self.train_epochs,
            self.discount,
            self.eps_clip,
            self.training
        )
        # remove previous log files
        if os.path.exists(os.path.join("logging", "PPO")):
            os.rmdir(os.path.join("logging", "PPO"))
        if os.path.exists(os.path.join("logging", "PPOAI")):
            os.rmdir(os.path.join("logging", "PPOAI"))
        
    def initialize(self, gameData, isPlayerOne):
        # set command center
        self.command = self.gateway.jvm.aiinterface.CommandCenter()
        # init structures
        self.key = self.gateway.jvm.struct.Key()
        self.frameData = self.gateway.jvm.struct.FrameData()
        # save game info
        self.player = isPlayerOne
        self.gameData = gameData
        self.character = self.gameData.getCharacterName(self.player)
        self.simulator = self.gameData.getSimulator()
        # set checkpoint file name
        self.checkpoint_name = "ppo_" + self.character + ".pt"
        # load model if necessary
        if os.path.exists(self.checkpoint_name):
            self.model.load(self.checkpoint_name)
            print("PPO Checkpoint Loaded")
        print("AI Initialized, Mode=" + ("training" if self.training else "playing"))
        # create session writer
        self.writer = SummaryWriter("logging/PPOAI")
        self.initialized = True
        return 0

    def getInformation(self, frameData, isControl):
        # whether AI can act
        self.isControl = isControl
        # update framedata
        self.frameData = frameData
        self.command.setFrameData(self.frameData, self.player)

    def processing(self):
        # if round end or just started, do not process
        if self.frameData.getEmptyFlag() or self.frameData.getRemainingTime() <= 0:
            return
        # if there is a skill not executed yet, skip
        if self.command.getSkillFlag():
            self.key = self.command.getSkillKey()
            return
        # if not in control, skip
        if not self.isControl:
            return
        if self.training:
            self.training_steps_count += 1
        # empty actions
        self.key.empty()
        self.command.skillCancel()
        # get observation
        state = self.observe()
        # get next action
        action_digit = self.model.action([state])
        action = self.actions.actions_map_useful[action_digit]
        print("Action: " + action)
        # get reward
        reward = self.getReward()
        self.reward_sum += reward
        self.writer.add_scalar("Reward", reward, self.sim_count)
        self.writer.add_scalar("Reward Accumulated", self.reward_sum, self.sim_count)
        self.sim_count += 1
        # if training, get next observation and train
        if self.training:
            self.model.update(reward, False)
            # if meet training step, train
            if self.training_steps_count % self.training_steps == 0:
                print("PPO Training")
                self.model.train()
        # execute action
        self.command.commandCall(action)

    def input(self):
        # return chosen action
        return self.key

    def close(self):
        # if is training, save current model
        if self.initialized:
            if self.training:
                self.model.save(self.checkpoint_name)
            self.writer.close()
        print("Game ended")

    def roundEnd(self, p1Hp, p2Hp, frames):
        # save round end info
        if self.training:
            self.model.update(self.getReward(), True)
            print("PPO Training")
            self.model.train()
            self.training_steps_count = 0
        print("Round Ended")
    
    def observe(self) -> List:
        """
        Observe current state, and create state info
        """
        me = self.frameData.getCharacter(self.player)
        opp = self.frameData.getCharacter(not self.player)
        obs = []
        # information of me
        obs.append(me.getCenterX()) # get position X
        obs.append(me.getCenterY()) # get position Y
        obs.append(me.getEnergy()) # get energy
        obs.append(me.getSpeedX()) # get horizontal speed
        obs.append(me.getSpeedY()) # get vertical speed
        obs.append(me.isFront()) # whether facing front
        obs.append(self.states_map[me.getState()]) # get state STAND / CROUCH/ AIR / DOWN
        # information of opponent
        obs.append(opp.getCenterX())
        obs.append(opp.getCenterY())
        obs.append(opp.getEnergy())
        obs.append(opp.getSpeedX())
        obs.append(opp.getSpeedY())
        obs.append(opp.isFront())
        obs.append(self.states_map[opp.getState()])
        return obs

    def getHPs(self) -> Tuple[int, int]:
        """
        Get HP info, self and enemy
        """
        me = self.frameData.getCharacter(self.player)
        opp = self.frameData.getCharacter(not self.player)
        return me.getHp(), opp.getHp()

    def getReward(self) -> int:
        """
        Get reward: change in HP(me) - change in HP(opponent)
        """
        me, opp = self.hp_me, self.hp_opp
        self.hp_me, self.hp_opp = self.getHPs()
        return (self.hp_me - me) - (self.hp_opp - opp)