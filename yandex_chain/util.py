import json

class YException(Exception):
    pass

class YAuth:
    def __init__(self, folder_id, api_key=None, iam_token=None):
        self.folder_id = folder_id
        self.api_key = api_key
        self.iam_token = iam_token

    @property
    def headers(self):
        if self.folder_id is not None and self.api_key is not None:
            return { 
                        "Authorization" : f"Api-key {self.api_key}",
                        "x-folder-id" : self.folder_id 
                   }
        if self.iam_token is not None:
            return {
                "Authorization" : f"Bearer {self.iam_token}",
                "x-folder-id" : self.folder_id 
            }

    @staticmethod
    def from_dict(js):
        if js.get('folder_id') is not None and js.get('api_key') is not None:
            return YAuth(js['folder_id'],api_key=js['api_key'])
        if js.get('iam_token') is not None:
            return YAuth(js['folder_id'], iam_token=js['iam_token'])
        raise YException("Cannot create valid authentication object: you need to provide folder_id and either iam token or api_key fields")

    @staticmethod
    def from_config_file(fn):
        with open(fn,'r',encoding='utf-8') as f:
            js = json.load(f)
        return YAuth.from_dict(js)

    @staticmethod
    def from_params(kwargs):
        if kwargs.get('config') is not None:
            return YAuth.from_config_file(kwargs['config'])
        return YAuth.from_dict(kwargs)        
