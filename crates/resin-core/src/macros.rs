#[macro_export]
macro_rules! arrvec {
    // This only works if the ArrayVec is the same size as the input array.
    ($elem:expr; $n:expr) => ({
        $crate::ArrayVec::from([$elem; $n])
    });
    // This just repeatedly calls `push`. I don't believe there's a concise way to count the number of expressions.
    ($($x:expr),*$(,)*) => ({
        // Allow an unused mut variable, since if the sequence is empty,
        // the vec will never be mutated.
        #[allow(unused_mut)] {
            let mut vec = $crate::arrayvec::ArrayVec::new();
            $(vec.push($x);)*
            vec
        }
    });
}
