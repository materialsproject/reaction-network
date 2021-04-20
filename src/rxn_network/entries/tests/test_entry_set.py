def test_from_entries(self):
    gibbs_entries = GibbsComputedEntry.from_entries(self.mp_entries)
    self.assertIsNotNone(gibbs_entries)

def test_from_pd(self):
    pd = PhaseDiagram(self.mp_entries)
    gibbs_entries = GibbsComputedEntry.from_pd(pd)
    self.assertIsNotNone(gibbs_entries)