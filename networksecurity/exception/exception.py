import sys
from networksecurity.logging import logger

class NetworkSecurityException(Exception):
    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(error_message)
        self.error_message = str(error_message)

        # Try to extract traceback info safely
        _, _, exc_tb = error_detail.exc_info()
        if exc_tb is not None:
            self.lineno = exc_tb.tb_lineno
            self.filename = exc_tb.tb_frame.f_code.co_filename
        else:
            # Case when exception is raised manually (no traceback)
            self.lineno = "N/A"
            self.filename = "Raised manually (no traceback)"

    def __str__(self):
        return (
            f"Error occurred in script: {self.filename} "
            f"at line number: {self.lineno} | Error message: {self.error_message}"
        )


if __name__ == "__main__":
    try:
        logger.logging.info("Testing NetworkSecurityException")
        # force an artificial error for demo
        raise ValueError("This is a test error.")
    except Exception as e:
        raise NetworkSecurityException(e, sys)
