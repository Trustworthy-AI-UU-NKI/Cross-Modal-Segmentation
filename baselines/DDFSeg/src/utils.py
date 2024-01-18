import random


def fake_image_pool(self, num_fakes, fake, fake_pool):

    if num_fakes < self._pool_size:
        fake_pool[num_fakes] = fake
        return fake
    else:
        p = random.random()
        if p > 0.5:
            random_id = random.randint(0, self._pool_size - 1)
            temp = fake_pool[random_id]
            fake_pool[random_id] = fake
            return temp
        else:
            return fake