"""Optional HTTP Basic access control for a trusted-network pilot.

Enable with both SAFESIGHT_AUTH_USER and SAFESIGHT_AUTH_PASSWORD. TLS must be
provided by a reverse proxy for remote use. Health exposes only liveness.
"""
import base64
import binascii
import os
import secrets
from urllib.parse import urlsplit

from starlette.responses import Response


class OperatorAuth:
    def __init__(self,app):
        self.app = app
        self.username = os.getenv('SAFESIGHT_AUTH_USER','')
        self.password = os.getenv('SAFESIGHT_AUTH_PASSWORD','')
        if bool(self.username)!=bool(self.password):
            raise ValueError('Configure both SAFESIGHT_AUTH_USER and SAFESIGHT_AUTH_PASSWORD')

    async def __call__(self,scope,receive,send):
        if scope['type'] not in {'http','websocket'}:
            return await self.app(scope,receive,send)
        headers = dict(scope.get('headers',[]))
        origin = headers.get(b'origin')
        if origin and (scope['type']=='websocket' or scope.get('method') in {'POST','PATCH','PUT','DELETE'}):
            try:
                same_origin = urlsplit(origin.decode('ascii')).netloc.lower()==headers.get(b'host',b'').decode('ascii').lower()
            except (ValueError,UnicodeError):
                same_origin = False
            if not same_origin:
                if scope['type']=='websocket':
                    await send({'type':'websocket.close','code':1008,'reason':'Origin not allowed'})
                else:
                    await Response('Origin not allowed',status_code=403)(scope,receive,send)
                return
        if not self.username or scope.get('path')=='/health':
            return await self.app(scope,receive,send)
        authorized = False
        try:
            kind,encoded = headers.get(b'authorization',b'').decode('ascii').split(' ',1)
            if kind.lower()=='basic':
                username,password = base64.b64decode(encoded,validate=True).decode('utf-8').split(':',1)
                authorized = secrets.compare_digest(username.encode(),self.username.encode()) & secrets.compare_digest(password.encode(),self.password.encode())
        except (ValueError,UnicodeError,binascii.Error):
            pass
        if not authorized:
            if scope['type']=='websocket':
                await send({'type':'websocket.close','code':1008,'reason':'Authentication required'})
            else:
                response = Response('Authentication required',status_code=401,
                                    headers={'WWW-Authenticate':'Basic realm="SafeSight", charset="UTF-8"',
                                             'Cache-Control':'no-store'})
                await response(scope,receive,send)
            return
        scope['safesight_operator'] = self.username
        await self.app(scope,receive,send)
