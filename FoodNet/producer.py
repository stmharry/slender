import tensorflow as tf

from .blob import Blob
from .util import DEBUG

class BaseProducer(object):
    @staticmethod
    def get_queue_enqueue(self, values, capacity, dtypes=[tf.float32], shapes=[None], auto_enqueue=False):
        queue = tf.FIFOQueue(capacity, dtypes=dtypes, shapes=shapes)
        enqueue = queue.enqueue_many(values)
        if auto_enqueue:
            queue_runner = tf.train.QueueRunner(queue, [enqueue])
            tf.train.add_queue_runner(queue_runner)
        return (queue, enqueue)


class QueueProducer(BaseProducer):
    CAPACITY = 1024

    def __init__(self, capacity=CAPACITY):
        self.capacity = capacity

    def blob(self, name='image', shape=None, dtype=tf.float32):
        self.placeholder = tf.placeholder(
            name=name,
            shape=shape,
            dtype=dtype)
        (self.queue, self.enqueue) = self.get_queue_enqueue(values=[self.placeholder], dtype=dtype, shape=shape, auto=False)
        image = self.queue.dequeue()
        image = DEBUG(image, name='QueueProducer.image', func=tf.shape)
        return Blob(images=image)

    def eval(self):

    def kwargs(self, image):
        return dict(
            feed_dict={self.placeholder: image},
            fetch=dict(queue_producer_enqueue=self.enqueue))


class FileProducer(BaseProducer):
    CAPACITY = 32
    NUM_TRAIN_INPUTS = 8
    NUM_TEST_INPUTS = 1
    SUBSAMPLE_SIZE = 64

    def __init__(self,
                 capacity=CAPACITY,
                 num_train_inputs=NUM_TRAIN_INPUTS,
                 num_test_inputs=NUM_TEST_INPUTS,
                 subsample_size=SUBSAMPLE_SIZE):

        self.capacity = capacity
        self.num_train_inputs = num_train_inputs
        self.num_test_inputs = num_test_inputs
        self.subsample_size = subsample_size

    def _blob(self,
              image_dir,
              num_inputs=1,
              subsample_divisible=True,
              check=False,
              shuffle=False):

        filename_list = list()
        classname_list = list()

        for class_name in META.class_names:
            class_dir = os.path.join(image_dir, class_name)
            for (file_dir, _, file_names) in os.walk(class_dir):
                for file_name in file_names:
                    if not file_name.endswith('.jpg'):
                        continue
                    if (hash(file_name) % self.subsample_size == 0) != subsample_divisible:
                        continue
                    filename_list.append(os.path.join(file_dir, file_name))
                    classname_list.append(class_name)

        label_list = map(META.class_names.index, classname_list)

        if check:
            num_file_list = list()
            for (num_file, filename) in enumerate(filename_list):
                print('\033[2K\rChecking image %d / %d' % (num_file + 1, len(filename_list)), end='')
                sp = subprocess.Popen(['identify', filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                (stdout, stderr) = sp.communicate()
                if stderr:
                    os.remove(filename)
                    print('\nRemove %s' % filename)
                else:
                    num_file_list.append(num_file)
                sys.stdout.flush()
            print('')

            filename_list = map(filename_list.__getitem__, num_file_list)
            label_list = map(label_list.__getitem__, num_file_list)

        images = list()
        labels = list()
        for num_input in xrange(num_inputs):
            if shuffle:
                perm = np.random.permutation(len(filename_list))
                filename_list = map(filename_list.__getitem__, perm)
                label_list = map(label_list.__getitem__, perm)

            filename_queue = self.get_queue_enqueue(filename_list, dtype=tf.string, shape=(), auto=True)[0]
            (key, value) = tf.WholeFileReader().read(filename_queue)
            image = tf.to_float(tf.image.decode_jpeg(value))

            label_queue = self.get_queue_enqueue(label_list, dtype=tf.int64, shape=(), auto=True)[0]
            label = label_queue.dequeue()

            images.append(image)
            labels.append(label)

        return Blob(images=images, labels=labels)

    def trainBlob(self, image_dir, check=True):
        return self._blob(
            image_dir,
            num_inputs=self.num_train_inputs,
            subsample_divisible=False,
            check=check,
            shuffle=True)

    def testBlob(self, image_dir, check=False):
        return self._blob(
            image_dir,
            num_inputs=self.num_test_inputs,
            subsample_divisible=True,
            check=check,
            shuffle=False)

    def kwargs(self):
        return dict()

