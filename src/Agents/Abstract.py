# abstract class for basic AI interface
# refer to: https://github.com/TeamFightingICE/FightingICE/blob/master/src/aiinterface/AIInterface.java
class AIInterface(object):
    def initialize(self, gameData, isPlayerOne):
        """
	    This method initializes AI, and it will be executed only once in the
	    beginning of each game.\\
	    Its execution will load the data that cannot be changed and load the flag
	    of the player's side ("Boolean player", `true` for P1 or
	    `false` for P2).\\
	    If there is anything that needs to be initialized, you had better do it
	    in this method.\\
	    It will return 0 when such initialization finishes correctly, otherwise
	    the error code.
        """
        pass
    
    def getInformation(self, frameData, isControl):
        """
        Gets information from the game status in each frame.\\
	    Such information is stored in the parameter frameData.\\
	    If `frameData.getRemainingTime()` returns a negative value, the
	    current round has not started yet.\\
	    When you use frameData received from getInformation(),\\
	    you must always check if the condition
	    `!frameData.getEmptyFlag() && frameData.getRemainingTime() > 0`
	    holds; otherwise, `NullPointerException` will occur.\\
	    You must also check the same condition when you use the CommandCenter
	    class.
        """
        pass

    def processing(self):
        """
        Processes the data from AI.\\
	    It is executed in each frame.
        """
        pass

    def input(self):
        """
        Receives a key input from AI.\\
	    It is executed in each frame and returns a value in the Key type.
        """
        pass

    def close(self):
        """
        Finalizes AI.\\
        It runs only once at the end of each game.
        """
        pass

    def roundEnd(self, p1Hp, p2Hp, frames):
        """
        Informs the result of each round.\\
	    It is called when each round ends.
        """
        pass

    def getScreenData(self, screenData):
        """
        Gets the screen information in each frame.
        """
        pass

    # This part is mandatory
    class Java:
        implements = ["aiinterface.AIInterface"]