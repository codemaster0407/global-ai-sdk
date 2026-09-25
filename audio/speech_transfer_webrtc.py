'''
WebRTC voice receiver.

A front-end (browser / mobile app) captures the microphone, creates a WebRTC
offer and POSTs it to /offer. This server answers, receives the audio track,
and writes it to data/audio_files/<session>.wav as 16 kHz mono PCM.

Run from the repo root:
    python -m audio.speech_transfer_webrtc                 # http://localhost:8080
    python -m audio.speech_transfer_webrtc --cert cert.pem --key key.pem   # https (needed for other devices)

Open http://localhost:8080 for a test client.
'''
import argparse
import asyncio
import os
import ssl
import uuid
import wave

from aiohttp import web
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from av import AudioResampler

ROOT = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(os.path.dirname(ROOT), 'data', 'audio_files')

SAMPLE_RATE = 16000  # 16 kHz mono s16 is what most speech-to-text models expect

ICE_SERVERS = [RTCIceServer(urls='stun:stun.l.google.com:19302')]

peer_connections = set()


async def consume_audio(track, session_id, on_audio=None):
    '''
    Read frames from the incoming track, resample to 16 kHz mono s16 and
    write them to a WAV file. `on_audio(pcm_bytes)` is an optional hook for
    live processing (e.g. streaming speech-to-text).
    '''
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    path = os.path.join(RECORDINGS_DIR, f'{session_id}.wav')
    resampler = AudioResampler(format='s16', layout='mono', rate=SAMPLE_RATE)

    with wave.open(path, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)

        print(f'[{session_id}] recording to {path}')
        while True:
            try:
                frame = await track.recv()
            except MediaStreamError:
                break

            for out in resampler.resample(frame):
                pcm = bytes(out.planes[0])[: out.samples * 2]
                wav.writeframes(pcm)
                if on_audio:
                    on_audio(pcm)

    print(f'[{session_id}] audio track ended, saved {path}')


async def offer(request):
    params = await request.json()
    session_id = uuid.uuid4().hex[:8]

    pc = RTCPeerConnection(RTCConfiguration(iceServers=ICE_SERVERS))
    peer_connections.add(pc)

    @pc.on('connectionstatechange')
    async def on_connectionstatechange():
        print(f'[{session_id}] connection state: {pc.connectionState}')
        if pc.connectionState in ('failed', 'closed'):
            await pc.close()
            peer_connections.discard(pc)

    @pc.on('track')
    def on_track(track):
        print(f'[{session_id}] received {track.kind} track')
        if track.kind == 'audio':
            asyncio.ensure_future(consume_audio(track, session_id))

    await pc.setRemoteDescription(RTCSessionDescription(sdp=params['sdp'], type=params['type']))
    await pc.setLocalDescription(await pc.createAnswer())

    # aiortc gathers ICE candidates before returning, so the answer is complete
    return web.json_response({
        'sdp': pc.localDescription.sdp,
        'type': pc.localDescription.type,
        'session_id': session_id,
    })


async def index(request):
    return web.FileResponse(os.path.join(ROOT, 'static', 'webrtc_client.html'))


@web.middleware
async def cors(request, handler):
    # Allow a front-end hosted on a different origin to call /offer
    if request.method == 'OPTIONS':
        response = web.Response()
    else:
        response = await handler(request)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


async def on_shutdown(app):
    await asyncio.gather(*(pc.close() for pc in peer_connections))
    peer_connections.clear()


def create_app():
    app = web.Application(middlewares=[cors])
    app.on_shutdown.append(on_shutdown)
    app.router.add_get('/', index)
    app.router.add_post('/offer', offer)
    app.router.add_route('OPTIONS', '/offer', offer)
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WebRTC voice receiver')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--cert', help='SSL certificate file (enables https)')
    parser.add_argument('--key', help='SSL key file')
    args = parser.parse_args()

    ssl_context = None
    if args.cert:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(args.cert, args.key)

    web.run_app(create_app(), host=args.host, port=args.port, ssl_context=ssl_context)
