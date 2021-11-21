enum State {
    Start,
    Passthrough,
    NewShard,
    Finish,
    Exhausted
}

pub trait Sharder<T: ?Sized>: Sized {
    fn shard(&self, subj: &T) -> Option<usize>;
    fn shard_start(&self, shard: usize) -> T;
    fn shard_end(&self, shard: usize) -> T;
}

pub struct BracketedChunks<I, S>
  where I: Iterator, S: Sharder<I::Item>
{
    state: State,
    sharder: S,
    candidate: Option<I::Item>,
    source: I,
}

impl<I, S> Iterator for BracketedChunks<I, S>
where
    I: Iterator,
    S: Sharder<I::Item>
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let mut candidate = None;
        std::mem::swap(&mut candidate, &mut self.candidate);
        let (state, cur, next) = match (&self.state, candidate) {
            (State::Start, None) => {
                let sharder = &self.sharder;
                let first = self.source.find(|v| sharder.shard(&v).is_some()).unwrap();
                (State::Passthrough, Some(self.sharder.shard_start(self.sharder.shard(&first).unwrap())), Some(first))
            },
            (State::Start, Some(_)) =>
                panic!("Invalid state, Start with candidate!"),
            (State::Passthrough, Some(c)) => match (self.sharder.shard(&c), self.source.next()) {
                // Cur is valid, and we have a next
                (Some(shard), Some(next)) => match self.sharder.shard(&next) {
                    Some(next_shard) if shard == next_shard =>
                        (State::Passthrough, Some(c), Some(next)),
                    Some(_) =>
                        (State::NewShard, Some(c), Some(next)),
                    None => 
                        (State::Finish, Some(c), Some(self.sharder.shard_end(shard)))
                },
                // Cur is valid, but no next
                (Some(shard), None) =>
                    (State::Finish, Some(c), Some(self.sharder.shard_end(shard))),
                // Cur is invalid
                (None, _) =>
                    panic!("Invalid state, passthrough with out-of-range value!")
            },
            (State::Passthrough, None) =>
                panic!("Invalid state, passthrough without candidate!"),
            (State::NewShard, Some(c)) => 
                (State::Passthrough, Some(self.sharder.shard_start(self.sharder.shard(&c).unwrap())), Some(c)),
            (State::NewShard, None) =>
                panic!("Invalid state, new shard without candidate!"),
            (State::Finish, c) =>
                (State::Exhausted, c, None),
            (State::Exhausted, _) =>
                (State::Exhausted, None, None)
        };
        self.candidate = next;
        self.state = state;
        cur
    }
}

pub trait Bracketed<S>: Iterator
  where S: Sharder<Self::Item>
{
    fn bracketed_chunks(self, sharder: S) -> BracketedChunks<Self, S>
        where Self: Sized, S: Sized
    {
        BracketedChunks {
            state: State::Start,
            sharder: sharder,
            candidate: None,
            source: self
        }
    }
}

impl<I: Iterator, S> Bracketed<S> for I
    where S: Sharder<I::Item> {}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools;

    fn dbgIter<I, T>(it: I) -> impl Iterator<Item=T> where I: Iterator<Item=T>, T: std::fmt::Debug {
        let collected = it.collect::<Vec<_>>();
        dbg!(&collected);
        collected.into_iter()
    }

    #[test]
    fn partitions_things() {
        struct IntSharder {
            min: u32,
            max: u32,
            span: u32
        }
        impl Sharder<u32> for IntSharder {
            fn shard(&self, val: &u32) -> Option<usize> {
                match val {
                    v if *v < self.min => None,
                    v if *v > self.max => None,
                    v => Some((v/self.span) as usize)
                }
            }
            fn shard_start(&self, shard: usize) -> u32 { shard as u32*self.span }
            fn shard_end(&self, shard: usize) -> u32 { (shard as u32+1)*self.span }
        }

        let bracketed = dbgIter((5..50).step_by(10).bracketed_chunks(IntSharder { min: 10, max: 40, span: 10 }));
        assert!(itertools::equal([10,15,20,25,30,35,40], bracketed));
    }
}
