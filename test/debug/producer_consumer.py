# Specify the number of consumer and producer threads
from threading import Thread, Lock
import time

numproducers = 2
runtime = 8
producer_freqs = [1.5, 1.5]
consumer_delay = 0.5

# Create a lock so that only one thread writes to the console at a time
safeprint = Lock()

# Create a queue object
data_list = [None for _ in range(numproducers)]
locks = [Lock() for _ in range(len(data_list))]
start = time.time()


# Function called by the producer thread
def producer(prod_id):
    num_msgs = int(runtime / producer_freqs[prod_id] + 1)
    last_set = None
    for i in range(num_msgs):
        # Simulate a delay
        if last_set:
            delta = producer_freqs[prod_id] - (time.perf_counter() - last_set)
            if delta > 0:
                time.sleep(delta)

        # Put a String on the queue
        # with queue_locks[idnum]:
        data = f"[producer id={prod_id}, count={i}]"
        with locks[prod_id]:
            data_list[prod_id] = data

        with safeprint:
            print(f"{time.time() - start:.3f}: producer #{prod_id} set => {data}")

        last_set = time.perf_counter()


# Function called by the consumer threads
def consumer():
    # Create an infinite loop
    while True:
        while any(d is None for d in data_list):
            pass

        values = []
        for i in range(len(data_list)):
            with locks[i]:
                values.append(data_list[i])
                data_list[i] = None

        # Acquire a lock on the console
        time.sleep(consumer_delay)
        with safeprint:
            # Print the data created by the producer thread
            print(f"{time.time() - start:.3f}: consumer got => {values}")


if __name__ == "__main__":
    assert numproducers == len(producer_freqs)

    # Create consumers
    tc = Thread(target=consumer, daemon=True)
    tc.start()

    # Create producers
    producers = [Thread(target=producer, args=(i,)) for i in range(numproducers)]
    for p in producers:
        p.start()

    for p in producers:
        p.join()

    # Exit the program
    print("Main thread exit")
