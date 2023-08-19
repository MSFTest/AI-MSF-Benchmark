class Mail:
    """ Dummy class to print messages without sending e-mails"""
    def __init__(self,mailaddress):
        pass
    def msg(self,msg):
        print(msg)
    def finalize(self,success,benchmark,sha_key,mailaddress=None):
        if success:
            print("Results for %s (system: %s) sucessfully created" % (benchmark,sha_key))
        else:
            print("Creating results for %s (system: %s) failed" % (benchmark,sha_key))

