enum State {
    Start,
    Passthrough,
    Exhausted
}

#[derive(Debug,PartialEq,Clone)]
pub enum ShardResult<T, U> {
  Start(U, T),
  Item(U, T),
  End(U, T)
}

pub trait Sharder<T: ?Sized>: Sized {
    fn shard(&self, subj: &T) -> Option<usize>;
    /*
    fn shard_start(&self, shard: usize) -> T;
    fn shard_end(&self, shard: usize) -> T;
    */
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
    type Item = ShardResult<I::Item, usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut candidate = None;
        std::mem::swap(&mut candidate, &mut self.candidate);
        let (state, cur, next) = match (&self.state, candidate) {
            (State::Start, None) => {
                let sharder = &self.sharder;
                match self.source.find(|v| sharder.shard(&v).is_some()) {
                  Some(first) => (
                    State::Passthrough,
                    Some(ShardResult::Start(self.sharder.shard(&first).unwrap(), first)),
                    self.source.next()
                  ),
                  None => (State::Exhausted, None, None)
                }
            },
            (State::Start, Some(_)) =>
                panic!("Invalid state, Start with candidate!"),

            (State::Passthrough, Some(c)) => match (self.sharder.shard(&c), self.source.next()) {
                // Cur is valid, and we have a next
                (Some(shard), Some(next)) => match self.sharder.shard(&next) {
                    Some(next_shard) if shard == next_shard =>
                        (State::Passthrough, Some(ShardResult::Item(shard, c)), Some(next)),
                    Some(_) =>
                        (State::Passthrough, Some(ShardResult::Start(shard, c)), Some(next)),
                    None => 
                        (State::Exhausted, Some(ShardResult::End(shard, c)), None)
                },
                // Cur is valid, but no next
                (Some(shard), None) =>
                    (State::Exhausted, Some(ShardResult::End(shard, c)), None),
                // Cur is invalid
                (None, _) =>
                    panic!("Invalid state, passthrough with out-of-range value!")
            },
            (State::Passthrough, None) =>
                (State::Exhausted, None, None),

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
            /*
            fn shard_start(&self, shard: usize) -> u32 { shard as u32*self.span }
            fn shard_end(&self, shard: usize) -> u32 { (shard as u32+1)*self.span }
            */
        }

        let bracketed = dbgIter((5..50).step_by(10)
          .bracketed_chunks(IntSharder { min: 10, max: 40, span: 10 }));
        assert!(itertools::equal([
            ShardResult::Start(1, 15),
            ShardResult::Start(2, 25),
            ShardResult::End(3, 35)
        ], bracketed));
    }
}
