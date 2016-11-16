import numpy as np
import Queue
import threading
import tensorflow as tf
import time

from slender.producer import PlaceholderProducer as Producer
from slender.processor import BaseProcessor
from slender.net import OnlineNet as Net
from slender.util import latest_working_dir

DEBUG = False
if DEBUG:
    import tensorflow as tf
    from tensorflow.python.client import timeline

from env import WORKING_DIR_ROOT

WORKING_DIR = latest_working_dir(WORKING_DIR_ROOT)
QUEUE_SIZE = 1024
BATCH_SIZE = 64
NET_DIM = 256
GPU_FRAC = 1.0
TIMEOUT_FN = lambda size: 0.02 + 0.01 * (1 - ((BATCH_SIZE - 2 * float(size)) / BATCH_SIZE) ** 2)


class Processor(BaseProcessor):
    def __init__(self,
                 net_dim,
                 batch_size=64):

        super(Processor, self).__init__(
            net_dim=net_dim,
            batch_size=batch_size,
        )

    def preprocess_single(self, content):
        images = [BaseProcessor._decode(content)] * self.num_duplicates
        images = self.mean_subtraction(images)
        images = self.set_shape(images)
        image = tf.pack(images)
        return image


class Task(object):
    def __init__(self, inputs):
        self.event = threading.Event()
        self.inputs = inputs
        self._input_slices = None
        self.outputs = None
        self._output_slice_list = []
        self.offset = 0
        self.size = 0

    def is_finished(self):
        return self.offset == len(self.inputs)

    def prepare_input(self, size):
        size = min(
            size,
            len(self.inputs) - self.offset,
        )
        self._input_slices = self.inputs[self.offset:self.offset + size]
        self.offset += size
        self.size = size

    def register_output(self, outputs):
        self._output_slice_list.append(outputs)

    def prepare_output(self):
        self.outputs = np.concatenate(self._output_slice_list)
        self.event.set()

    def eval(self, factory):
        factory.queue.put(self)
        self.event.wait()


class Factory(object):
    def __init__(self,
                 working_dir=WORKING_DIR,
                 queue_size=QUEUE_SIZE,
                 batch_size=BATCH_SIZE,
                 net_dim=NET_DIM,
                 gpu_frac=GPU_FRAC,
                 timeout_fn=TIMEOUT_FN):

        producer = Producer(
            working_dir=working_dir,
            batch_size=batch_size,
        )
        processor = Processor(
            net_dim=net_dim,
            batch_size=batch_size,
        )
        net = Net(
            working_dir=working_dir,
            num_classes=producer.num_classes,
            gpu_frac=gpu_frac,
        )
        blob = producer.blob().funcs([
            processor.preprocess,
            net.forward,
            processor.postprocess,
        ])

        queue = Queue.Queue(maxsize=queue_size)
        thread = threading.Thread(target=self.run)

        self.__dict__.update(locals())
        net.init()
        thread.start()

    def run(self):
        tasks = []
        total_size = 0

        while True:  # serve_forever
            while total_size < self.batch_size:
                try:
                    task = self.queue.get(timeout=None if total_size == 0 else self.timeout_fn(total_size))
                    task.get_timestamp = time.time()
                except Queue.Empty:
                    break
                else:
                    task.prepare_input(self.batch_size - total_size)
                    tasks.append(task)
                    total_size += task.size

            content = np.concatenate([task.inputs for task in tasks])

            if DEBUG:
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            else:
                options = None
                run_metadata = None

            start = time.time()
            blob_val = self.blob.eval(
                sess=self.net.sess,
                feed_dict={self.producer.content: content},
                options=options,
                run_metadata=run_metadata,
            )
            duration = time.time() - start

            print('Factory.run: batch_size={}, duration={:.4f}'.format(len(content), duration))
            outputs = np.split(blob_val.predictions, np.cumsum([task.size for task in tasks]))

            if DEBUG:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                with open('./trace/trace_{}.json'.format(len(content)), 'w') as f:
                    f.write(trace.generate_chrome_trace_format())

            total_size = 0
            for (task, output) in zip(tasks, outputs):
                task.register_output(output)

                if task.is_finished():
                    task.prepare_output()
                    tasks.remove(task)
                    self.queue.task_done()
                else:
                    task.prepare_input(self.batch_size - total_size)
                    total_size += task.size
