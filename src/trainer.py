from decimal import Decimal

import torch
import torch.nn.utils as utils
from tqdm import tqdm

import utility


class Trainer:
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.optimizer.schedule()
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
                '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        loss_exits = None
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, idx_scale)
            loss = self.loss(sr, hr)
            if loss_exits is None and self.args.multi_exit:
                loss_exits = [0 for _ in range(len(sr))]
            if self.args.multi_exit:
                temp_loss = loss[1:]
                for i in range(len(temp_loss)):
                    loss_exits[i] += temp_loss[i] / self.args.print_every
                loss = loss[0]
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(self.model.parameters(), self.args.gclip)
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                if loss_exits:
                    exits_loss_str = ""
                    exits_loss_str += "E0" + ": %.4f" % loss_exits[0].item()
                    for i in range(1, len(loss_exits)):
                        exits_loss_str += " E" + str(i) + ": %.4f" % loss_exits[i].item()
                    self.ckp.write_log('[{}/{}]\t{}\t{}\t{:.1f}+{:.1f}s'.format(
                            (batch + 1) * self.args.batch_size,
                            len(self.loader_train.dataset),
                            self.loss.display_loss(batch), exits_loss_str,
                            timer_model.release(), timer_data.release()))
                    loss_exits = None
                else:
                    self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                            (batch + 1) * self.args.batch_size,
                            len(self.loader_train.dataset),
                            self.loss.display_loss(batch),
                            timer_model.release(), timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch() + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results:
            self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                # eval_acc = 0
                eval_accs = []
                for lr, hr, filename, _ in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)

                    if self.args.multi_exit:
                        srs = self.model(lr, idx_scale)
                        sr_num = len(srs)
                        for i in range(sr_num):
                            sr_filename = filename[0] + '_' + str(i + 1)
                            sr = utility.quantize(srs[i], self.args.rgb_range)
                            save_list = [sr]

                            if len(eval_accs) <= i:
                                eval_accs.append(0)

                            eval_accs[i] += utility.calc_psnr(sr, hr, scale,
                                    self.args.rgb_range, dataset=d,
                                    force_y=self.args.force_y)
                            if i == sr_num - 1:
                                self.ckp.log[
                                    -1, idx_data, idx_scale] += utility.calc_psnr(
                                        sr, hr, scale, self.args.rgb_range,
                                        dataset=d, force_y=self.args.force_y)
                                save_list.extend([lr, hr])

                            if self.args.save_results:
                                self.ckp.save_results(d, sr_filename,
                                                      save_list, scale)
                    else:
                        sr = self.model(lr, idx_scale)
                        sr = utility.quantize(sr, self.args.rgb_range)

                        save_list = [sr]
                        # self.ckp.log[-1, idx_data, idx_scale]
                        self.ckp.log[
                            -1, idx_data, idx_scale] += utility.calc_psnr(sr,
                                hr, scale, self.args.rgb_range, dataset=d,
                                force_y=self.args.force_y)
                        if self.args.save_gt:
                            save_list.extend([lr, hr])

                        if self.args.save_results:
                            self.ckp.save_results(d, filename[0], save_list,
                                                  scale)

                # if self.args.multi_exit:
                #     max_eval_acc = 0
                #     for i in range(len(eval_accs)):
                #         if eval_accs[i] > max_eval_acc:
                #             max_eval_acc = eval_accs[i]
                #     eval_acc = max_eval_acc

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                # self.ckp.log[-1, idx_data, idx_scale] = eval_acc / len(d)
                best = self.ckp.log.max(0)
                if self.args.multi_exit:
                    output_str = ""
                    for i in range(len(eval_accs)):
                        eval_accs[i] = eval_accs[i] / len(self.loader_test)
                        output_str += (
                                    "PSNR " + str(i) + ": %.3f " % eval_accs[i])
                    self.ckp.write_log(
                            '[{} x{}]\t{} (Best: {:.3f} @epoch {})'.format(
                                    d.dataset.name, scale, output_str,
                                    best[0][idx_data, idx_scale],
                                    best[1][idx_data, idx_scale] + 1))
                else:
                    self.ckp.write_log(
                            '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                                    d.dataset.name, scale,
                                    self.ckp.log[-1, idx_data, idx_scale],
                                    best[0][idx_data, idx_scale],
                                    best[1][idx_data, idx_scale] + 1))
            if not self.args.test_only:
                for idx_scale, scale in enumerate(self.scale):
                    self.ckp.save_scale(self, epoch, scale, is_best=(
                                best[1][idx_data, idx_scale] + 1 == epoch))

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log('Total: {:.2f}s\n'.format(timer_test.toc()),
                refresh=True)

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs
