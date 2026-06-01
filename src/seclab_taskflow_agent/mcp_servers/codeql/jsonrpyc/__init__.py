
from __future__ import annotations

__all__: list[str] = []

import json
import os
import re
import sys
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol, TypeAlias

# package infos
from .__meta__ import (  # noqa
    __author__,
    __contact__,
    __copyright__,
    __credits__,
    __doc__,
    __email__,
    __license__,
    __status__,
    __version__,
)

Callback: TypeAlias = Callable[[Exception | None, Any | None], None]


class InputStream(Protocol):
    def fileno(self) -> int: ...

    @property
    def closed(self) -> bool: ...

    def isatty(self) -> bool: ...

    def tell(self) -> int: ...

    def seek(self, offset: int, whence: int = os.SEEK_SET, /) -> int: ...

    def readline(self) -> str: ...

    def readlines(self) -> list[str]: ...


class OutputStream(Protocol):
    def fileno(self) -> int: ...

    @property
    def closed(self) -> bool: ...

    def write(self, b: str) -> int: ...

    def flush(self) -> None: ...


class Spec:
    """
    This class wraps methods that create JSON-RPC 2.0 compatible string representations of
    request, response and error objects. All methods are class members, so you might never want to
    create an instance of this class, but rather use the methods directly:

    .. code-block:: python

        Spec.request("my_method", 18)  # the id is optional
        # => '{"jsonrpc":"2.0","method":"my_method","id": 18}'

        Spec.response(18, "some_result")
        # => '{"jsonrpc":"2.0","id":18,"result":"some_result"}'

        Spec.error(18, -32603)
        # => '{"jsonrpc":"2.0","id":18,"error":{"code":-32603,"message":"Internal error"}}'
    """

    @classmethod
    def check_id(cls, id_: str | int | None, *, allow_empty: bool = False) -> None:
        """
        Value check for *id_* entries. When *allow_empty* is *True*, *id_* is allowed to be *None*.
        Raises a *TypeError* when *id_* is neither an integer nor a string.

        :param id_: The id to check.
        :param allow_empty: Whether *id_* is allowed to be *None*.
        :return: None.
        :raises TypeError: When *id_* is invalid.
        """
        if (id_ is not None or not allow_empty) and not isinstance(id_, (int, str)):
            raise TypeError(f"id must be an integer or string, got {id_} ({type(id_)})")

    @classmethod
    def check_method(cls, method: str, /) -> None:
        """
        Value check for *method* entries. Raises a *TypeError* when *method* is not a string.

        :param method: The method to check.
        :return: None.
        :raises TypeError: When *method* is invalid.
        """
        if not isinstance(method, str):
            raise TypeError(f"method must be a string, got {method} ({type(method)})")

    @classmethod
    def check_code(cls, code: int, /) -> None:
        """
        Value check for *code* entries. Raises a *TypeError* when *code* is not an integer, or a
        *KeyError* when there is no :py:class:`RPCError` subclass registered for that *code*.

        :param code: The error code to check.
        :return: None.
        :raises TypeError: When *code* is invalid.
        """
        try:
            get_error(code)
        except Exception:
            raise TypeError(f"invalid error code, got {code} ({type(code)})")

    @classmethod
    def request(
        cls,
        method: str,
        /,
        id_: str | int | None = None,
        *,
        params: dict[str, Any] | None = None,
    ) -> str:
        """
        Creates the string representation of a request that calls *method* with optional *params*
        which are encoded by ``json.dumps``. When *id_* is *None*, the request is considered a
        notification.

        :param method: The method to call.
        :param id_: The id of the request.
        :param params: The parameters of the request.
        :return: The request string.
        :raises RPCInvalidRequest: When *method* or *id_* are invalid.
        :raises RPCParseError: When *params* could not be encoded.
        """
        try:
            cls.check_method(method)
            cls.check_id(id_, allow_empty=True)
        except Exception as e:
            raise RPCInvalidRequest(str(e))

        # start building the request string
        req = f'{{"jsonrpc":"2.0","method":"{method}"'

        # add the id when given
        if id_ is not None:
            # encode string ids
            if isinstance(id_, str):
                id_ = json.dumps(id_)
            req += f',"id":{id_}'

        # add parameters when given
        if params is not None:
            try:
                req += f',"params":{json.dumps(params)}'
            except Exception as e:
                raise RPCParseError(str(e))

        # end the request string
        req += "}"

        return req

    @classmethod
    def response(cls, id_: str | int | None, result: Any, /) -> str:
        """
        Creates the string representation of a respone that was triggered by a request with *id_*.
        A *result* is required, even if it is *None*.

        :param id_: The id of the request that triggered this response.
        :param result: The result of the request.
        :return: The response string.
        :raises RPCInvalidRequest: When *id_* is invalid.
        :raises RPCParseError: When *result* could not be encoded.
        """
        try:
            cls.check_id(id_)
        except Exception as e:
            raise RPCInvalidRequest(str(e))

        # encode string ids
        if isinstance(id_, str):
            id_ = json.dumps(id_)

        # build the response string
        try:
            res = f'{{"jsonrpc":"2.0","id":{id_},"result":{json.dumps(result)}}}'
        except Exception as e:
            raise RPCParseError(str(e))

        return res

    @classmethod
    def error(
        cls,
        id_: str | int | None,
        code: int,
        *,
        data: Any | None = None,
    ) -> str:
        """
        Creates the string representation of an error that occured while processing a request with
        *id_*. *code* must lead to a registered :py:class:`RPCError`. *data* might contain
        additional, detailed error information and is encoded by ``json.dumps`` when set.

        :param id_: The id of the request that triggered this error.
        :param code: The error code.
        :param data: Additional error data.
        :return: The error string.
        :raises RPCInvalidRequest: When *id_* or *code* are invalid.
        :raises RPCParseError: When *data* could not be encoded.
        """
        try:
            cls.check_id(id_)
            cls.check_code(code)
        except Exception as e:
            raise RPCInvalidRequest(str(e))

        # build the inner error data
        message = get_error(code).title  # type: ignore[union-attr]
        err_data = f'{{"code":{code},"message":"{message}"'

        # insert data when given
        if data is not None:
            try:
                err_data += f',"data":{json.dumps(data)}}}'
            except Exception as e:
                raise RPCParseError(str(e))
        else:
            err_data += "}"

        # encode string ids
        if isinstance(id_, str):
            id_ = json.dumps(id_)

        # start building the error string
        err = f'{{"jsonrpc":"2.0","id":{id_},"error":{err_data}}}'

        return err


class RPC:
    """
    The main class of *jsonrpyc*. Instances of this class wrap an input stream *stdin* and an output
    stream *stdout* in order to communicate with other services. A service is not even forced to be
    written in Python as long as it strictly implements the JSON-RPC 2.0 specification. RPC
    instances may wrap a *target* object. By means of a :py:class:`Watchdog` instance, incoming
    requests are routed to methods of this object whose result might be sent back as a response.
    The watchdog instance is created but not started yet, when *watch* is not *True*.

    :param target: The target object to wrap.
    :param stdin: The input stream.
    :param stdout: The output stream.
    :param watch: Whether to start the watchdog.
    :param watch_kwargs: Additional keyword arguments for the watchdog.

    Example implementation:

    *server.py*

    .. code-block:: python

        import jsonrpyc


        class MyTarget(object):
            def greet(self, name):
                return f"Hi, {name}!"


        jsonrpc.RPC(MyTarget())

    *client.py*

    .. code-block:: python

        import jsonrpyc
        from subprocess import Popen, PIPE

        p = Popen(["python", "server.py"], stdin=PIPE, stdout=PIPE)
        rpc = jsonrpyc.RPC(stdout=p.stdin, stdin=p.stdout)

        # non-blocking remote procedure call with callback and js-like signature
        def cb(err, res=None):
            if err:
                throw err
            print(f"callback got: {res}")

        rpc("greet", args=("John",), callback=cb)

        # cb is called asynchronously which prints
        # => "callback got: Hi, John!"

        # blocking remote procedure call with 0.1s polling
        print(rpc("greet", args=("John",), block=0.1))
        # => "Hi, John!"

        # shutdown the process
        p.stdin.close()
        p.stdout.close()
        p.terminate()
        p.wait()

    .. py:attribute:: target

        The wrapped target object. Might be *None* when no object is wrapped, e.g. for the *client*
        RPC instance.

    .. py:attribute:: stdin

        The input stream, re-opened with ``"rb"``.

    .. py:attribute:: stdout

        The output stream, re-opened with ``"wb"``.

    .. py:attribute:: watch

        The :py:class:`Watchdog` instance that optionally watches *stdin* and dispatches incoming
        requests.
    """

    EMPTY_RESULT = object()

    def __init__(
        self,
        target: Any | None = None,
        stdin: InputStream | None = None,
        stdout: OutputStream | None = None,
        *,
        watch: bool = True,
        watch_kwargs: dict[str, Any] | None = None,
        method_handlers: dict[str, Any] = {},
    ) -> None:
        super().__init__()

        # method handlers map methods to f(params) handlers
        self.method_handlers = method_handlers

        # the wrapped target object
        self.target = target

        # open input stream
        if stdin is None:
            stdin = sys.stdin
        self.original_stdin = stdin
        self.stdin = open(stdin.fileno(), "rb")

        # open output stream
        if stdout is None:
            stdout = sys.stdout
        self.original_stdout = stdout
        self.stdout = open(stdout.fileno(), "wb")

        # other attributes
        self._i = -1
        self._callbacks: dict[int, Callback] = {}
        self._results: dict[int, Any] = {}

        # create and optionally start the watchdog
        watch_kwargs = watch_kwargs or {}
        watch_kwargs["start"] = watch
        watch_kwargs.setdefault("daemon", target is None)
        self.watchdog = Watchdog(self, **watch_kwargs)

    def stop_watchdog(self) -> None:
        watchdog = getattr(self, "watchdog", None)
        self.watchdog = None
        if watchdog:
            watchdog.stop()
            watchdog.join()

    def __del__(self) -> None:
        self.stop_watchdog()

    def __call__(self, *args, **kwargs) -> None:
        """
        Shorthand for :py:meth:`call`.
        """
        return self.call(*args, **kwargs)

    def call(
        self,
        method: str,
        args: tuple[Any, ...] = (),
        kwargs: dict | None = None,
        *,
        callback: Callback | None = None,
        block: int = 0,
        timeout: float = 0,
        params: dict | None = None,
    ) -> int:
        """
        Performs an actual remote procedure call by writing a request representation (a string) to
        the output stream. The remote RPC instance uses *method* to route to the actual method to
        call with *args* and *kwargs*.

        When *callback* is set, it will be called with the result of the remote call. When *block*
        is larger than *0*, the calling thread is blocked until the result is received. In this
        case, *block* will be the poll interval, emulating synchronuous return value behavior.
        When both *callback* is *None* and *block* is *0* or smaller, the request is considered a
        notification and the remote RPC instance will not send a response.

        If *timeout* is not zero, raise TimeoutError after *timeout* seconds with no response.

        :param method: The method to call.
        :param args: The positional arguments of the method.
        :param kwargs: The keyword arguments of the method.
        :param params: Optional raw parameter override.
        :param callback: The callback function.
        :param block: The poll interval in seconds.
        :param timeout: The timeout in seconds.
        :return: the RPC Request id for this call.
        :raises TimeoutError: When the request times out.
        """
        starting_time = time.monotonic()

        # default kwargs
        if kwargs is None:
            kwargs = {}

        # check if the call is a notification
        is_notification = callback is None and block <= 0

        # create a new id for requests expecting a response
        id = -1
        if not is_notification:
            self._i += 1
            id = self._i

        # register the callback
        if callback is not None:
            self._callbacks[id] = callback

        # store an empty result for the meantime
        if block > 0:
            self._results[id] = self.EMPTY_RESULT

        # create the request
        params = params if params else {"args": args, "kwargs": kwargs}
        req = Spec.request(method, id_=id, params=params)
        print(f"-> {req}")
        msg = f"Content-Length: {len(req)}\r\n\r\n{req}"
        self._write(msg)

        # blocking return value behavior
        if block > 0:
            while True:
                if self._results[id] != self.EMPTY_RESULT:
                    result = self._results[id]
                    del self._results[id]
                    if isinstance(result, Exception):
                        raise result
                    return result

                if timeout:
                    elapsed = time.monotonic() - starting_time
                    if elapsed > timeout:
                        raise TimeoutError("RPC Request timed out")

                time.sleep(block)
        return id

    def _handle(self, msg: str) -> None:
        """
        Handles an incoming *line* and dispatches the parsed object to the request, response, or
        error handlers.

        :param msg: The incoming jsonrpc message.
        :return: None.
        """
        print(f"<- {msg}")
        obj = json.loads(msg)

        # dispatch to the correct handler
        if "method" in obj:
            # request
            # self._handle_request(obj)
            if self.method_handlers:
                self._handle_method(obj)
            else:
                # this does class routing, be careful!
                self._handle_request(obj)
        elif "error" not in obj:
            # response
            self._handle_response(obj)
        else:
            # error
            self._handle_error(obj)

    def _handle_method(self, req: dict[str, Any]) -> None:
        """
        Handles an incoming request *req*. When it containes an id, a response or error is sent
        back.

        :param req: The incoming request.
        :return: None.
        """
        try:
            method = req["method"]
            if method not in self.method_handlers:
                raise ValueError(f"No handler defined for method: {method}")
            result = self.method_handlers[method](req["params"])
            if "id" in req:
                res = Spec.response(req["id"], result)
                self._write(res)
        except Exception as e:
            if "id" in req:
                if isinstance(e, RPCError):
                    err = Spec.error(req["id"], e.code, data=e.data)
                else:
                    err = Spec.error(req["id"], -32603, data=str(e))
                self._write(err)

    def _handle_request(self, req: dict[str, Any]) -> None:
        """
        Handles an incoming request *req*. When it containes an id, a response or error is sent
        back.

        :param req: The incoming request.
        :return: None.
        """
        try:
            method = self._route(req["method"])
            result = method(*req["params"]["args"], **req["params"]["kwargs"])
            if "id" in req:
                res = Spec.response(req["id"], result)
                self._write(res)
        except Exception as e:
            if "id" in req:
                if isinstance(e, RPCError):
                    err = Spec.error(req["id"], e.code, data=e.data)
                else:
                    err = Spec.error(req["id"], -32603, data=str(e))
                self._write(err)

    def _handle_response(self, res: dict[str, Any]) -> None:
        """
        Handles an incoming successful response *res*. Blocking calls are resolved and registered
        callbacks are invoked with the first error argument being set to *None*.

        :param res: The incoming response.
        :return: None.
        """
        # set the result
        if res["id"] in self._results:
            self._results[res["id"]] = res["result"]

        # lookup and invoke the callback
        if res["id"] in self._callbacks:
            callback = self._callbacks[res["id"]]
            del self._callbacks[res["id"]]
            callback(None, res["result"])

    def _handle_error(self, res: dict[str, Any]) -> None:
        """
        Handles an incoming failed response *res*. Blocking calls throw an exception and
        registered callbacks are invoked with an exception and the second result argument set to
        *None*.

        :param res: The incoming error response.
        :return: None.
        """
        # extract the error and create an actual error instance to raise
        err = res["error"]
        error = get_error(err["code"])(err.get("data", err["message"]))

        # set the error
        if res["id"] in self._results:
            self._results[res["id"]] = error

        # lookup and invoke the callback
        if res["id"] in self._callbacks:
            callback = self._callbacks[res["id"]]
            del self._callbacks[res["id"]]
            callback(error, None)

    def _route(self, method: str) -> Any:
        """
        Returnes the method of the wrapped target object to be called when *method* is requested.

        :param method: The method to route to.
        :return: The method to call.
        :raises RPCMethodNotFound: When the method is not found.

        Example:

        .. code-block:: python

            MyClassB(object):
                def foo(self):
                    return 123

            MyClassA(object):
                def __init__(self):
                    self.b = MyClassB()

                def bar(self):
                    return "test"

            rpc = RPC(MyClassA())

            rpc._route("bar")
            # => <bound method MyClassA.bar ...>

            rpc._route("b.foo")
            # => <bound method MyClassB.foo ...>
        """
        # recursively traverse target attributes
        obj = self.target
        for part in method.split("."):
            if not hasattr(obj, part):
                break
            obj = getattr(obj, part)
        else:
            return obj

        raise RPCMethodNotFound(data=method)

    def _write(self, s: str) -> None:
        """
        Writes a string *s* to the output stream.

        :param s: The string to write.
        :return: None.
        """
        self.stdout.write(bytearray(f"{s}", "utf-8"))
        self.stdout.flush()


class Watchdog(threading.Thread):
    """
    This class represents a thread that watches the input stream of an :py:class:`RPC` instance for
    incoming content and dispatches requests to it.

    :param rpc: The :py:class:`RPC` instance to watch.
    :param name: The thread's name.
    :param interval: The polling interval of the run loop.
    :param daemon: The thread's daemon flag.
    :param start: Whether to start the thread immediately.

    .. py:attribute:: rpc

        The :py:class:`RPC` instance.

    .. py:attribute:: name

        The thread's name.

    .. py:attribute:: interval

        The polling interval of the run loop.

    .. py:attribute:: daemon

        The thread's daemon flag.
    """

    def __init__(
        self,
        rpc: RPC,
        name: str = "watchdog",
        interval: float = 0.1,
        daemon: bool = False,
        start: bool = True,
    ) -> None:
        super().__init__()

        # store attributes
        self.rpc = rpc
        self.name = name
        self.interval = interval
        self.daemon = daemon

        # register a stop event
        self._stopper = threading.Event()

        if start:
            self.start()

    def start(self) -> None:
        """
        Starts the thread's activity.

        :return: None.
        """
        super().start()

    def join(self, *args, **kwargs) -> None:
        """
        Join the thread back

        :return: None.
        """
        super().join(*args, **kwargs)

    def stop(self) -> None:
        """
        Stops the thread's activity.

        :return: None.
        """
        self._stopper.set()

    def run(self) -> None:
        """
        The main run loop of the watchdog thread. Reads the input stream of the :py:class:`RPC`
        instance and dispatches incoming content to it.

        :return: None.
        """
        # reset the stop event
        self._stopper.clear()

        # stop here when stdin is not set or closed
        if self.rpc.stdin is None or self.rpc.stdin.closed:
            return

        while not self._stopper.is_set():
            # stop when stdin is closed
            if self.rpc.stdin.closed:
                break

            # Keep linter happy
            if self.rpc.original_stdin and self.rpc.original_stdin.closed:  # type: ignore[attr-defined]
                break

            try:
                line = self.rpc.stdin.readline()
            except OSError:
                line = None

            if line:
                decoded_line = line.decode("utf-8").strip()
                match = re.search(r"^Content-Length:\s*([0-9]+)", decoded_line, re.IGNORECASE)
                debug = False
                if match:
                    if debug:
                        print(f"Incoming message header: {decoded_line}")
                    content_length = abs(int(match.group(1)))
                    _ = self.rpc.stdin.readline()
                    if debug:
                        print(f"Grabbing {content_length} bytes from stdin")
                    msg = b""
                    while len(msg) != content_length:
                        msg += self.rpc.stdin.read(content_length - len(msg))
                    decoded_msg = msg.decode("utf-8").strip()
                    if debug:
                        print(f"Incoming jsonrpc message: {decoded_msg}")
                    self.rpc._handle(decoded_msg)
                else:
                    raise ValueError(f"Don't know how to handle: {decoded_line}")
            else:
                self._stopper.wait(self.interval)


class RPCError(Exception):
    """
    Base class for RPC errors.

    :param data: Additional error data.

    .. py:attribute:: code_range

        The range of error codes that this error class is responsible for.

    .. py:attribute:: code

        The error code of this error.

    .. py:attribute:: title

        The title of this error.

    .. py:attribute:: message

        The message of this error, i.e., ``"<title> (<code>)[, data: <data>]"``.

    .. py:attribute:: data

        Additional data of this error. Setting the data attribute will also change the message
        attribute.
    """

    code_range: tuple[int, int]
    code: int
    title: str

    @classmethod
    def is_code_range(cls, code: Any) -> bool:
        """
        Returns *True* when *code* is a valid error code range, i.e., a tuple of two integers where
        the first integer is less or equal to the second integer.

        :param code: The code to check.
        :return: Whether *code* is a valid error code range.
        """
        return (
            isinstance(code, tuple) and len(code) == 2 and all(isinstance(i, int) for i in code) and code[0] <= code[1]
        )

    def __init__(self, data: str | None = None) -> None:
        # build the error message
        message = f"{self.title} ({self.code})"
        if data is not None:
            message += f", data: {data}"
        self.message = message

        super().__init__(message)

        self.data = data

    def __str__(self) -> str:
        return self.message


error_map_code: dict[int, type[RPCError]] = {}
error_map_code_range: dict[tuple[int, int], type[RPCError]] = {}


def register_error(cls: type[RPCError]) -> type[RPCError]:
    """
    Decorator that registers a new RPC error derived from :py:class:`RPCError`. The purpose of
    error registration is to have a mapping of error codes/code ranges to error classes for faster
    lookups during error creation.

    .. code-block:: python

        @register_error
        class MyCustomRPCError(RPCError):
            code_range = (lower_bound, upper_bound)  # both inclusive
            code = code_range[0]  # default code when used as is
            title = "My custom error"
    """
    if not issubclass(cls, RPCError):
        raise TypeError(f"'{cls}' is not a subclass of RPCError")

    # check duplicates
    if cls.code in error_map_code:
        raise AttributeError(f"duplicate RPC error code {cls.code}")
    if cls.code_range in error_map_code_range:
        raise AttributeError(f"duplicate RPC error code range {cls.code_range}")

    # register
    error_map_code[cls.code] = cls
    error_map_code_range[cls.code_range] = cls

    return cls


def get_error(code: int) -> type[RPCError]:
    """
    Returns the RPC error class that was previously registered to *code*. A ``ValueError`` is raised
    if no error class was found for *code*.

    :param code: The error code to look up.
    :return: The error class.
    :raises ValueError: When no error class was found for *code*.
    """
    if code in error_map_code:
        return error_map_code[code]

    for (lower, upper), cls in error_map_code_range.items():
        if lower <= code <= upper:
            return cls

    raise ValueError(f"unknown error code '{code}' ({type(code)})")


@register_error
class RPCParseError(RPCError):
    code_range = (-32700, -32700)
    code = code_range[0]
    title = "Parse error"


@register_error
class RPCInvalidRequest(RPCError):
    code_range = (-32600, -32600)
    code = code_range[0]
    title = "Invalid Request"


@register_error
class RPCMethodNotFound(RPCError):
    code_range = (-32601, -32601)
    code = code_range[0]
    title = "Method not found"


@register_error
class RPCInvalidParams(RPCError):
    code_range = (-32602, -32602)
    code = code_range[0]
    title = "Invalid params"


@register_error
class RPCInternalError(RPCError):
    code_range = (-32603, -32603)
    code = code_range[0]
    title = "Internal error"


@register_error
class RPCServerError(RPCError):
    code_range = (-32099, -32000)
    code = code_range[0]  # default code when used as is
    title = "Server error"
