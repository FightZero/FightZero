# implementation of PPO AI
import os
import glob
import torch
from datetime import datetime
from typing import Tuple
from torch.utils.tensorboard.writer import SummaryWriter
from .Abstract import AIInterface
from ..Algorithms.PPOBM import PPO
from ..Utils.ActionsBM import Actions

class PPOAI(AIInterface):
    def __init__(self, gateway, gameRounds=2, train=False, frameSkip=False):
        self.gateway = gateway
        # set whether in training mode
        self.training = train
        self.training_steps = 4000
        self.training_steps_count = 0
        self.frame_skip = frameSkip
        # set parameters
        self.actions = Actions()
        self.state_dimensions = 143 # NOTE: set correct number of dimensions here
        self.action_dimensions = self.actions.count # 56
        self.lr_actor = 1e-5
        self.lr_critic = 3e-4
        self.train_epochs = 120
        self.discount = 0.99
        self.eps_clip = 0.2
        self.batchsize = 128
        self.max_grad_norm = 0.2
        self.reward_sum = 0
        self.reward_eps = 0
        if os.path.exists('sim_count.txt'):
            with open('sim_count.txt','r') as f:
                self.sim_count=int(f.read())
        else:
            self.sim_count = 0
        self.num_win = 0
        self.num_lose = 0
        self.game_count = 0
        self.round_count = 0
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
        if len(os.listdir(os.path.join("logging", "PPOAI")))>0:
            list_of_files = glob.glob(os.path.join("logging", "PPOAI")+'\\*')
            self.writer_session = os.path.basename(max(list_of_files, key=os.path.getctime))
        else:
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
        print('me:',self.gameData.getAiName(self.player),'\n')
        self.simulator = self.gameData.getSimulator()
        self.justStarted = True
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
            self.justStarted = True
            return
        if self.frame_skip:
            # if there is a skill not executed yet, skip
            if self.command.getSkillFlag():
                self.key = self.command.getSkillKey()
                return
            if not self.isControl:
                return
            # empty actions
            self.key.empty()
            self.command.skillCancel()
        if self.training:
            self.training_steps_count += 1
        # get observation
        state = self.observe()
        # get next action
        action_digit = self.model.action(state)
        action = self.actions.actions[action_digit]
        print("Action: " + action)
        # get reward
        reward = self.getReward()
        self.reward_sum += reward
        self.reward_eps += reward
        self.writer.add_scalar("PPOAI/Reward", reward, self.sim_count)
        self.writer.add_scalar("PPOAI/Reward Accumulated", self.reward_sum, self.sim_count)
        self.sim_count += 1
        with open('sim_count.txt','w') as f:
            f.write(int(self.sim_count))
        # if training, get next observation and train
        if self.training:
            if not self.justStarted:
                self.model.update(reward, False)
            else:
                self.justStarted = False
            # if meet training step, train
            if self.training_steps_count % self.training_steps == 0:
                print("PPO Training")
                self.model.train(self.writer)
                self.training_steps_count = 0
        # execute action
        self.command.commandCall(action)
        if not self.frame_skip:
            self.key = self.command.getSkillKey()

    def input(self):
        # return chosen action
        return self.key

    def close(self):
        self.game_count += 1
        # if is training, save current model
        if self.training and self.game_count >= self.game_total:
            print("PPO Training")
            self.model.train(self.writer)
        self.model.save(self.checkpoint_name)
        self.writer.add_scalar("PPOAI/Num Win", self.num_win, self.game_count)
        self.writer.add_scalar("PPOAI/Num Lose", self.num_lose, self.game_count)
        self.writer.close()
        if self.game_count==1:
            with open("win_lose.txt", "w") as f:
                f.write(str(self.num_win)+'\n')
                f.write(str(self.num_lose)+'\n')
            f.close()
        else:
            with open("win_lose.txt", "a") as f:
                f.write(str(self.num_win)+'\n')
                f.write(str(self.num_lose)+'\n')
            f.close()
        self.num_win, self.num_lose = 0, 0
        self.reward_sum = 0.0
        print("Game ended")

    def roundEnd(self, p1Hp, p2Hp, frames):
        # update win/lose count
        # self.model.action(self.observe())
        if self.player:
            if p1Hp >= p2Hp:
                self.num_win += 1
                # self.model.update(1.0, True)
                # self.reward_sum += 1.0
                # self.writer.add_scalar("PPOAI/Reward", 1.0, self.sim_count)
                print("Round End, Win!")
            else:
                self.num_lose += 1
                # self.model.update(-1.0, True)
                # self.reward_sum += -1.0
                # self.writer.add_scalar("PPOAI/Reward", -1.0, self.sim_count)
                print("Round End, Lose!")
            reward=self.getReward()
            self.reward_sum += reward
            self.writer.add_scalar("PPOAI/Reward", reward, self.sim_count)
            self.model.update(reward, True)
        else:
            if p1Hp <= p2Hp:
                self.num_win += 1
                # self.model.update(1.0, True)
                # self.reward_sum += 1.0
                # self.writer.add_scalar("PPOAI/Reward", 1.0, self.sim_count)
                print("Round End, Win!")
            else:
                self.num_lose += 1
                # self.model.update(-1.0, True)
                # self.reward_sum += -1.0
                # self.writer.add_scalar("PPOAI/Reward", -1.0, self.sim_count)
                print("Round End, Lose!")
            reward=self.getReward()
            self.reward_sum += reward
            self.writer.add_scalar("PPOAI/Reward", reward, self.sim_count)
            self.model.update(reward, True)
        self.writer.add_scalar("PPOAI/Reward Accumulated", self.reward_sum, self.sim_count)
        self.writer.add_scalar("PPOAI/Reward Episodic", self.reward_eps, self.round_count)
        self.reward_eps = 0
        self.round_count += 1
        self.sim_count += 1
        # reset health
        self.hp_me, self.hp_opp = 400, 400
        print("Round Ended")
    
    def observe(self) -> torch.Tensor:
        """
        Observe current state, and create state info
        """
        # nextFrameData = self.simulator.simulate(self.frameData, self.player, None, None, 17)
        obs = self.extract(self.frameData)
        # obs.extend(self.extract(nextFrameData))
        return obs

    def extract(self, frameData) -> torch.Tensor:
        """
        Extract observations
        """
        me = frameData.getCharacter(self.player)
        opp = frameData.getCharacter(not self.player)
        obs = []
        # information of me
        obs.append(me.getHp() / 400.0)
        obs.append(me.getEnergy() / 300.0) # get energy
        obs.append((me.getLeft() + me.getRight()) * 0.5 / 960.0) # get position X
        obs.append((me.getBottom() + me.getTop()) * 0.5 / 640.0) # get position Y
        obs.append(int(me.getSpeedX() >= 0.0))
        obs.append(abs(me.getSpeedX()) / 15.0) # get horizontal speed
        obs.append(int(me.getSpeedY() >= 0.0))
        obs.append(abs(me.getSpeedY()) / 28.0) # get vertical speed
        actionMe = [0.0] * self.actions.count
        actionMe[me.getAction().ordinal()] = 1.0
        obs.extend(actionMe)
        obs.append(me.getRemainingFrame() / 70.0) # remaining frames to back to normal
        # information of opponent
        obs.append(opp.getHp() / 400.0)
        obs.append(opp.getEnergy() / 300.0)
        obs.append((opp.getLeft() + opp.getRight()) * 0.5 / 960.0)
        obs.append((opp.getBottom() + opp.getTop()) * 0.5 / 640.0)
        obs.append(int(opp.getSpeedX() >= 0.0))
        obs.append(abs(opp.getSpeedX()) / 15.0)
        obs.append(int(opp.getSpeedY() >= 0.0))
        obs.append(abs(opp.getSpeedY()) / 28.0)
        actionOpp = [0.0] * self.actions.count
        actionOpp[opp.getAction().ordinal()] = 1.0
        obs.extend(actionOpp)
        obs.append(opp.getRemainingFrame() / 70.0)
        # remaining time
        obs.append(frameData.getFramesNumber() / 3600.0)
        # for attacks
        attMe = frameData.getProjectilesByP1() if self.player else frameData.getProjectilesByP2()
        attOpp = frameData.getProjectilesByP2() if self.player else frameData.getProjectilesByP1()
        # num=sum(1 for item in enumerate(attMe))
        num=2
        attMeObs = [0.0] * 3 * num
        attOppObs = [0.0] * 3 * num
        for i, attack in enumerate(attMe):
            if i * 3 + 2<sum(1 for item in enumerate(attMe)):
                hitarea = attack.getCurrentHitArea()
                attMeObs[i * 3] = attack.getHitDamage() / 200.0
                attMeObs[i * 3 + 1] = (hitarea.getLeft() + hitarea.getRight()) * 0.5 / 960.0
                attMeObs[i * 3 + 2] = (hitarea.getBottom() + hitarea.getTop()) * 0.5 / 640.0
        for i, attack in enumerate(attOpp):
            if i * 3 + 2<sum(1 for item in enumerate(attOpp)):
                hitarea = attack.getCurrentHitArea()
                attOppObs[i * 3] = attack.getHitDamage() / 200.0
                attOppObs[i * 3 + 1] = (hitarea.getLeft() + hitarea.getRight()) * 0.5 / 960.0
                attOppObs[i * 3 + 2] = (hitarea.getBottom() + hitarea.getTop()) * 0.5 / 640.0
        obs.extend(attMeObs)
        obs.extend(attOppObs)
        obs = torch.clamp(torch.FloatTensor([obs]), 0.0, 1.0)
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
        # return ((self.hp_me - me) - (self.hp_opp - opp))