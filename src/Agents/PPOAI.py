# implementation of PPO AI
import os
from datetime import datetime
from typing import List, Tuple
from torch.utils.tensorboard.writer import SummaryWriter
from .Abstract import AIInterface
from ..Algorithms.PPO import PPO
from ..Utils.Actions import Actions

class PPOAI(AIInterface):
    def __init__(self, gateway, gameRounds=2, train=False):
        self.gateway = gateway
        # set whether in training mode
        self.training = train
        # self.training_steps = 3e3
        # self.training_steps_count = 0
        # set parameters
        self.actions = Actions()
        self.state_dimensions = 147 # NOTE: set correct number of dimensions here
        self.action_dimensions = self.actions.count # 56
        self.lr_actor = 1e-4
        self.lr_critic = 3e-4
        self.train_epochs = 100
        self.discount = 0.99
        self.eps_clip = 0.2
        self.batchsize = 128
        self.max_grad_norm = 0.5
        self.reward_sum = 0
        self.sim_count = 0
        self.num_win = 0
        self.num_lose = 0
        self.game_count = 0
        self.game_total = gameRounds
        self.hp_me, self.hp_opp = 400, 400 # NOTE: default 400 each side
        # create model
        self.model = PPO(
            self.state_dimensions,
            self.action_dimensions,
            self.lr_actor, self.lr_critic,
            self.train_epochs,
            self.discount,
            self.eps_clip,
            self.batchsize,
            self.max_grad_norm,
            self.training
        )
        # set session name
        self.writer_session = datetime.now().strftime("%b-%d_%H-%M-%S")
        # remove previous log files
        # if os.path.exists(os.path.join("logging", "PPOAI")):
        #     shutil.rmtree(os.path.join("logging", "PPOAI"))
        
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
        if os.path.exists(self.checkpoint_name) and self.game_count <= 0:
            self.model.load(self.checkpoint_name)
            print("PPO Checkpoint Loaded")
        # create session writer
        self.writer = SummaryWriter("logging/PPOAI/" + self.writer_session)
        print("AI Initialized, Mode=" + ("training" if self.training else "playing"))
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
        # if not self.isControl:
        #     
        # if self.training:
        #     self.training_steps_count += 1
        # empty actions
        self.key.empty()
        self.command.skillCancel()
        # get observation
        state = self.observe()
        # get next action
        action_digit = self.model.action([state])
        action = self.actions.actions[action_digit]
        print("Action: " + action)
        # get reward
        reward = self.getReward()
        self.reward_sum += reward
        self.writer.add_scalar("PPOAI/Reward", reward, self.sim_count)
        self.writer.add_scalar("PPOAI/Reward Accumulated", self.reward_sum, self.sim_count)
        self.sim_count += 1
        # if training, get next observation and train
        if self.training:
            self.model.update(reward, False)
            # # if meet training step, train
            # if self.training_steps_count % self.training_steps == 0:
            #     print("PPO Training")
            #     self.model.train(self.writer)
            #     self.training_steps_count = 0
        # execute action
        self.command.commandCall(action)

    def input(self):
        # return chosen action
        return self.key

    def close(self):
        self.game_count += 1
        # if is training, save current model
        if self.training:
            print("PPO Training")
            self.model.train(self.writer)
            self.model.save(self.checkpoint_name)
        self.writer.add_scalar("PPOAI/Win Count", self.num_win, self.game_count)
        self.writer.add_scalar("PPOAI/Lose Count", self.num_lose, self.game_count)
        self.writer.close()
        self.num_win, self.num_lose = 0, 0
        print("Game ended")

    def roundEnd(self, p1Hp, p2Hp, frames):
        # update win/lose count
        if self.player:
            if p1Hp >= p2Hp:
                self.num_win += 1
                self.model.update(1.0, True)
                self.reward_sum += 1.0
                self.writer.add_scalar("PPOAI/Reward", 1.0, self.sim_count)
                print("Round End, Win!")
            else:
                self.num_lose += 1
                self.model.update(-1.0, True)
                self.reward_sum += -1.0
                self.writer.add_scalar("PPOAI/Reward", -1.0, self.sim_count)
                print("Round End, Lose!")
        else:
            if p1Hp <= p2Hp:
                self.num_win += 1
                self.model.update(1.0, True)
                self.reward_sum += 1.0
                self.writer.add_scalar("PPOAI/Reward", 1.0, self.sim_count)
                print("Round End, Win!")
            else:
                self.num_lose += 1
                self.model.update(-1.0, True)
                self.reward_sum += -1.0
                self.writer.add_scalar("PPOAI/Reward", -1.0, self.sim_count)
                print("Round End, Lose!")
        self.writer.add_scalar("PPOAI/Reward Accumulated", self.reward_sum, self.sim_count)
        self.sim_count += 1
        # reset health
        self.hp_me, self.hp_opp = 400, 400
        # reset rewards
        self.reward_sum = 0.0
        print("Round Ended")
    
    def observe(self) -> List:
        """
        Observe current state, and create state info
        """
        # nextFrameData = self.simulator.simulate(self.frameData, self.player, None, None, 17)
        obs = self.extract(self.frameData)
        # obs.extend(self.extract(nextFrameData))
        return obs

    def extract(self, frameData) -> List:
        """
        Extract observations
        """
        me = frameData.getCharacter(self.player)
        opp = frameData.getCharacter(not self.player)
        obs = []
        # information of me
        obs.append(me.getHp() / 400.0)
        obs.append((me.getLeft() + me.getRight()) * 0.5) # get position X
        obs.append((me.getBottom() + me.getTop()) * 0.5) # get position Y
        obs.append(me.getEnergy() / 300.0) # get energy
        obs.append(abs(me.getSpeedX())) # get horizontal speed
        obs.append(abs(me.getSpeedY())) # get vertical speed
        obs.append(me.getHitCount()) # hit count
        obs.append(me.getRemainingFrame()) # remaining frames to back to normal
        obs.append(int(me.isFront())) # whether facing front
        obs.append(int(me.isHitConfirm())) # hit count
        obs.append(me.getState().ordinal()) # get state STAND / CROUCH/ AIR / DOWN
        # information of opponent
        obs.append(opp.getHp() / 400.0)
        obs.append((opp.getLeft() + opp.getRight()) * 0.5)
        obs.append((opp.getBottom() + opp.getTop()) * 0.5)
        obs.append(opp.getEnergy() / 300.0)
        obs.append(abs(opp.getSpeedX()))
        obs.append(abs(opp.getSpeedY()))
        obs.append(opp.getHitCount())
        obs.append(opp.getRemainingFrame())
        obs.append(int(opp.isFront()))
        obs.append(int(opp.isHitConfirm()))
        obs.append(opp.getState().ordinal())
        # for attacks
        attMe = frameData.getProjectilesByP1() if self.player else frameData.getProjectilesByP2()
        attOpp = frameData.getProjectilesByP2() if self.player else frameData.getProjectilesByP1()
        attMeObs = [0.0] * 6
        attOppObs = [0.0] * 6
        for i, attack in enumerate(attMe):
            hitarea = attack.getCurrentHitArea()
            attMeObs[i * 3] = attack.getHitDamage()
            attMeObs[i * 3 + 1] = (hitarea.getLeft() + hitarea.getRight()) * 0.5
            attMeObs[i * 3 + 2] = (hitarea.getBottom() + hitarea.getTop()) * 0.5
        for i, attack in enumerate(attOpp):
            hitarea = attack.getCurrentHitArea()
            attOppObs[i * 3] = attack.getHitDamage()
            attOppObs[i * 3 + 1] = (hitarea.getLeft() + hitarea.getRight()) * 0.5
            attOppObs[i * 3 + 2] = (hitarea.getBottom() + hitarea.getTop()) * 0.5
        obs.extend(attMeObs)
        obs.extend(attOppObs)
        # remaining time
        obs.append(frameData.getFramesNumber() / 1000.0)
        # onehot action vector
        actionMe = [0.0] * self.actions.count
        actionMe[me.getAction().ordinal()] = 1.0
        obs.extend(actionMe)
        actionOpp = [0.0] * self.actions.count
        actionOpp[opp.getAction().ordinal()] = 1.0
        obs.extend(actionOpp)
        # obs.append(me.getAction().ordinal())
        # obs.append(opp.getAction().ordinal())
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
        return ((self.hp_me - me) - (self.hp_opp - opp)) / 400.0
        # return (self.hp_me - self.hp_opp) / 400.0