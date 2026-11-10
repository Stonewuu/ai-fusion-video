/** 表体是唯一滚动源，表头只改变显示位置，避免延迟的 scroll 事件反向拉回表体。 */
export function attachStoryboardHorizontalScroll(
  headerViewport: HTMLElement,
  headerContent: HTMLElement,
  bodyViewport: HTMLElement,
) {
  const synchronize = () => {
    headerContent.style.transform = `translateX(${-bodyViewport.scrollLeft}px)`;
  };

  const onHeaderWheel = (event: WheelEvent) => {
    const horizontalDelta = event.deltaX || (event.shiftKey ? event.deltaY : 0);
    if (horizontalDelta === 0) {
      return;
    }
    const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? bodyViewport.clientWidth : 1;
    const before = bodyViewport.scrollLeft;
    bodyViewport.scrollLeft += horizontalDelta * unit;
    synchronize();
    if (bodyViewport.scrollLeft !== before) {
      event.preventDefault();
    }
  };

  bodyViewport.addEventListener("scroll", synchronize, { passive: true });
  headerViewport.addEventListener("wheel", onHeaderWheel, { passive: false });
  const resizeObserver = new ResizeObserver(synchronize);
  resizeObserver.observe(bodyViewport);
  resizeObserver.observe(headerContent);
  synchronize();

  return () => {
    bodyViewport.removeEventListener("scroll", synchronize);
    headerViewport.removeEventListener("wheel", onHeaderWheel);
    resizeObserver.disconnect();
  };
}
