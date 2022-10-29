"""RP2 Attack for YOLO models."""

from adv_patch_bench.attacks.rp2 import rp2_base


class RP2AttackYOLO(rp2_base.RP2AttackModule):
    """RP2 Attack for YOLO models."""

    def _loss_func(self, adv_img, obj_class, metadata):
        """Compute loss for YOLO models."""
        # Compute logits, loss, gradients
        out, _ = self._core_model(adv_img, val=True)
        conf = out[:, :, 4:5] * out[:, :, 5:]
        conf, labels = conf.max(-1)
        if obj_class is not None:
            loss = 0
            # Loop over EoT batch
            for c, label in zip(conf, labels):
                c_l = c[label == obj_class]
                if c_l.size(0) > 0:
                    # Select prediction from box with max confidence and ignore
                    # ones with already low confidence
                    # loss += c_l.max().clamp_min(self.min_conf)
                    loss += c_l.clamp_min(self.min_conf).sum()
            loss /= self.num_eot
        else:
            # loss = conf.max(1)[0].clamp_min(self.min_conf).mean()
            loss = conf.clamp_min(self.min_conf).sum()
            loss /= self.num_eot
        return loss
