#! /usr/bin/env python


from tollan.utils.log import get_logger
from tollan.utils.fmt import pformat_dict
from dasha.web.extensions.ipc import ipc
from dasha.web.extensions.celery import get_celery_app
from .kidsviewdata import KidsViewData


class SharedKidsViewData(object):

    logger = get_logger()

    @staticmethod
    def data_key_from_info(info):
        return f"{info['interface']}-{info['Obsnum']}-{info['SubObsNum']}" \
               f"-{info['ScanNum']}"

    @classmethod
    def _get_datastore(cls, key):
        cls.logger.debug(f"get datastore key={key}")
        return ipc.get_or_create('cache', label=key)

    task_registry = ipc.get_or_create('cache', label='data_tasks')

    @classmethod
    def get_data_tasks(cls):
        tasks = cls.task_registry.get()
        if tasks is None:
            cls.task_registry.set(set())
            tasks = cls.task_registry.get()
        cls.logger.debug(f"get data tasks {tasks}")
        return tasks

    @classmethod
    def register_task(cls, info):
        cls.logger.debug(f"register data tasks for {pformat_dict(info)}")
        tasks = cls.get_data_tasks()
        tasks.add(cls.data_key_from_info(info))
        cls.task_registry.set(tasks)

    @classmethod
    def has_task(cls, info):
        tasks = cls.get_data_tasks()
        return cls.data_key_from_info(info) in tasks

    @classmethod
    def set_data(cls, info, data):
        cls.logger.debug(f"set data for {pformat_dict(info)} {data}")
        cls._get_datastore(cls.data_key_from_info(info)).set(data)

    @classmethod
    def get_data(cls, info):
        cls.logger.debug(f"get data for {pformat_dict(info)}")
        cls._get_datastore(cls.data_key_from_info(info)).get()

    @classmethod
    def get_all_data(cls):
        tasks = cls.get_data_tasks()
        cls.logger.debug(f"get all data from tasks {tasks}")
        result = dict()
        for key in tasks:
            cls.logger.debug(f"get data key {key}")
            data = cls._get_datastore(key).get()
            if data is None:
                cls.logger.debug(f"data {key} is not available")
                continue
            result[key] = data
        return result


celery = get_celery_app()


def _process_kids_file(info):
    logger = get_logger()
    logger.debug(f"process kids file {pformat_dict(info)}")

    def save(self):
        result = self.to_dict()
        logger.debug(
                f"save result for {pformat_dict(info)} {pformat_dict(result)}")
        SharedKidsViewData.set_data(info, result)

    result = KidsViewData(info, save=save)
    result.save()


if celery is not None:

    @celery.task
    def process_kids_file(info):
        _process_kids_file(info)


def request_viewer_data(info):
    logger = get_logger()
    if not SharedKidsViewData.has_task(info):
        try:
            # process_kids_file.delay(info)
            _process_kids_file(info)
        except Exception as e:
            logger.debug(
                    f"failed request viewer data"
                    f" {pformat_dict(info)}: {e}", exc_info=True)
            pass
        else:
            SharedKidsViewData.register_task(info)
            logger.debug(f"requested viewer data {pformat_dict(info)}")


def get_viewer_data():
    return SharedKidsViewData.get_all_data()
