



    def process_scst(self,seq):
        N, T = seq.size()
        sents = []
        for n in range(N):
            tokens = []
            for t in range(T):
                ix = seq[n, t].item()
                if ix == self.eos_token:
                    break
                tokens.append(ix)
            sents.append(tokens)
        return sents


    def forward_cap_scst(self, batch):

        loss_dict = {}
        batch_ids = batch['ids']
        self.eval()
        with torch.no_grad():
            evaluation_dict_greedy = self.generate_caption(batch, mode='greedy')  ### compute  reward baseline
        self.train()
        evaluation_dict_sample = self.generate_caption(batch, mode='sample')  ### compute  reward baseline

        generated_sequences_t_v_greedy = self.process_scst(evaluation_dict_greedy['generated_sequences_t_v'])
        generated_sequences_t_v_sample = self.process_scst(evaluation_dict_sample['generated_sequences_t_v'])
        logprobs_t_v_sample = evaluation_dict_sample['logprobs_t_v'] 

        reward_greedy = self.scorer(batch_ids, generated_sequences_t_v_greedy)
        reward_sample = self.scorer(batch_ids, generated_sequences_t_v_sample)

        self.update_alpha(reward_sample, reward_greedy)
        rewards = reward_sample - reward_greedy * self.get_alpha()
        rewards = torch.from_numpy(rewards).float().cuda()
        caption_loss_tv = self.reward_loss(evaluation_dict_sample['generated_sequences_t_v'], logprobs_t_v_sample, rewards)    
        loss_dict['caption_loss_tv'] = caption_loss_tv
        

        return loss_dict


    def reward_loss(self, seq, logP, rewards):
        mask = seq !=self.eos_token 
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)
        rewards = rewards.view(-1, 1).expand_as(logP)
        logP = torch.masked_select(logP, mask)
        rewards = torch.masked_select(rewards, mask)
        loss = torch.mean(-logP * rewards)
        return loss