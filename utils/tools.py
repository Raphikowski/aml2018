import logging


def init_logger(log_path, to_console=False):
    """Init the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the
    terminal is saved in a permanent file.

    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        if to_console:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(stream_handler)


class RunningAverage(object):

    def __init__(self, window):
        self.window = window
        self.values = []
        self.mean = 0

    def update(self, value):
        self.values.append(value)
        if len(self.values) > self.window:
            self.mean += (value - self.values.pop(0)) / self.window
        else:
            self.mean = sum(self.values) / len(self.values)

    def __call__(self):
        return self.mean


def find_lr(model, optimizer, dataloader, criterion, wd=0.001, start_lr=1e-8, end_lr=10., beta=0.98):

    num = len(dataloader) - 1
    mult = (end_lr / start_lr) ** (1/num)

    lr = start_lr
    optimizer.param_groups[0]['lr'] = lr

    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []


    for inputs, labels in tqdm(dataloader):
        batch_num += 1

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)

        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses

        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss

        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # AdamW
        for group in optimizer.param_groups:
            for param in group['params']:
                param.data = param.data.add(-wd * group['lr'], param.data)
        optimizer.step()

        #Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr

    return log_lrs, losses


def get_metrics(log_file):
    with open(log_file, 'r') as temp:
        lines = temp.readlines()

    # Get metrics
    train_dict = {'Step': [], 'Acc': [], 'Loss': []}
    eval_dict = {'Step': [], 'Acc': [], 'Loss': []}
    is_conf_line = True
    conf = ''

    for line in lines:
        info = None

        if '[TRAIN]' in line:
            is_conf_line = False
            items = line.split('[TRAIN]')[-1].split(';')
            for item in items:
                key, value = item.strip().split(':')
                train_dict[key].append(float(value))
        if '[EVAL]' in line:
            items = line.split('[EVAL]')[-1].split(';')
            for item in items:
                key, value = item.strip().split(':')
                eval_dict[key].append(float(value))

        if is_conf_line:
            if 'INFO:' in line:
                line = line.split('INFO:')[-1].strip() + '\n'
            conf += line

    return conf, train_dict, eval_dict
