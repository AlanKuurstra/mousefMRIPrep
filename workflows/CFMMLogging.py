import logging

class NipypeLoggerClass():
    def __init__(self):
        self.stdout_handler = logging.getLogger('nipype').handlers[0]
        self.workflow_logger = logging.getLogger('nipype.workflow')
    def info(self,msg):
        prev_level = self.stdout_handler.level
        self.stdout_handler.setLevel('INFO')
        self.workflow_logger.info(msg)
        self.stdout_handler.setLevel(prev_level)
    def warning(self,msg):
        prev_level = self.stdout_handler.level
        self.stdout_handler.setLevel('INFO')
        self.workflow_logger.warning(msg)
        self.stdout_handler.setLevel(prev_level)
    def error(self,msg):
        prev_level = self.stdout_handler.level
        self.stdout_handler.setLevel('INFO')
        self.workflow_logger.error(msg)
        self.stdout_handler.setLevel(prev_level)

NipypeLogger = NipypeLoggerClass()