"""D2 的连接清理辅助函数；关闭连接不会删除集合或文档。"""


def close_database(database):
    """优先调用 close()，否则按上下文管理协议调用 __exit__()。"""
    close = getattr(database, "close", None)
    if callable(close):
        close()
        return
    exit_method = getattr(database, "__exit__", None)
    if callable(exit_method):
        exit_method(None, None, None)
